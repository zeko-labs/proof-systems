//! This module contains the implementation of the polynomial commitment scheme
//! called the Inner Product Argument (IPA) as described in [Efficient
//! Zero-Knowledge Arguments for Arithmetic Circuits in the Discrete Log
//! Setting](https://eprint.iacr.org/2016/263)

use crate::{
    commitment::{
        b_poly, b_poly_coefficients, combine_commitments, shift_scalar, squeeze_challenge,
        squeeze_prechallenge, BatchEvaluationProof, CommitmentCurve, EndoCurve,
    },
    error::CommitmentError,
    hash_map_cache::HashMapCache,
    utils::combine_polys,
    BlindedCommitment, PolyComm, PolynomialsToCombine, SRS as SRSTrait,
};
use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger, Field, One, PrimeField, UniformRand, Zero};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations, Radix2EvaluationDomain as D,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use blake2::{Blake2b512, Digest};
use groupmap::GroupMap;
use mina_poseidon::{sponge::ScalarChallenge, FqSponge};
use o1_utils::{
    field_helpers::{inner_prod, pows},
    math,
};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{cmp::min, iter::Iterator, ops::AddAssign};

#[serde_as]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(bound = "G: CanonicalDeserialize + CanonicalSerialize")]
pub struct SRS<G> {
    /// The vector of group elements for committing to polynomials in
    /// coefficient form.
    #[serde_as(as = "Vec<o1_utils::serialization::SerdeAs>")]
    pub g: Vec<G>,

    /// A group element used for blinding commitments.
    #[serde_as(as = "o1_utils::serialization::SerdeAs")]
    pub h: G,

    /// Commitments to Lagrange bases, per domain size.
    #[serde(skip)]
    pub lagrange_bases: HashMapCache<usize, Vec<PolyComm<G>>>,
}

impl<G> PartialEq for SRS<G>
where
    G: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.g == other.g && self.h == other.h
    }
}

#[inline(always)]
fn scalar_num_bits<F: PrimeField>() -> usize {
    F::MODULUS_BIT_SIZE as usize
}

#[inline(always)]
fn extract_window_u64(limbs: &[u64], bit_offset: usize, width: usize) -> usize {
    let limb_idx = bit_offset / 64;
    let bit_idx = bit_offset % 64;

    if limb_idx >= limbs.len() {
        return 0;
    }

    let lo = limbs[limb_idx] >> bit_idx;
    let hi = if bit_idx > 0 && limb_idx + 1 < limbs.len() {
        limbs[limb_idx + 1] << (64 - bit_idx)
    } else {
        0
    };

    let mask = (1usize << width) - 1;
    ((lo | hi) as usize) & mask
}

pub fn endos<G: CommitmentCurve>() -> (G::BaseField, G::ScalarField)
where
    G::BaseField: PrimeField,
{
    let endo_q: G::BaseField = mina_poseidon::sponge::endo_coefficient();
    let endo_r = {
        let potential_endo_r: G::ScalarField = mina_poseidon::sponge::endo_coefficient();
        let t = G::generator();
        let (x, y) = t.to_coordinates().unwrap();
        let phi_t = G::of_coordinates(x * endo_q, y);
        if t.mul(potential_endo_r) == phi_t.into_group() {
            potential_endo_r
        } else {
            potential_endo_r * potential_endo_r
        }
    };
    (endo_q, endo_r)
}

fn point_of_random_bytes<G: CommitmentCurve>(map: &G::Map, random_bytes: &[u8]) -> G
where
    G::BaseField: Field,
{
    // Pack in bit representation.
    const N: usize = 31;
    let extension_degree = G::BaseField::extension_degree() as usize;

    let mut base_fields = Vec::with_capacity(N * extension_degree);

    for base_count in 0..extension_degree {
        let mut bits = [false; 8 * N];
        let offset = base_count * N;
        for i in 0..N {
            for j in 0..8 {
                bits[8 * i + j] = (random_bytes[offset + i] >> j) & 1 == 1;
            }
        }

        let n =
            <<G::BaseField as Field>::BasePrimeField as PrimeField>::BigInt::from_bits_be(&bits);
        let t = <<G::BaseField as Field>::BasePrimeField as PrimeField>::from_bigint(n)
            .expect("packing code has a bug");
        base_fields.push(t)
    }

    let t = G::BaseField::from_base_prime_field_elems(base_fields).unwrap();

    let (x, y) = map.to_group(t);
    G::of_coordinates(x, y).mul_by_cofactor()
}

impl<G: CommitmentCurve> SRS<G>
where
    G::ScalarField: PrimeField,
{
    pub fn fixed_bases_with_h(&self, padding: usize) -> Vec<G> {
        let mut bases = Vec::with_capacity(self.g.len() + 1 + padding);
        bases.push(self.h);
        bases.extend(self.g.iter().copied());
        bases.extend(std::iter::repeat(G::zero()).take(padding));
        bases
    }

    pub fn fixed_base_msm(
        &self,
        fixed_bases: &[G],
        scalars: &[G::ScalarField],
        window_bits: usize,
    ) -> G::Group {
        assert_eq!(fixed_bases.len(), scalars.len());

        if fixed_bases.is_empty() {
            return G::Group::zero();
        }

        let scalar_bits = scalar_num_bits::<G::ScalarField>();
        let num_windows = scalar_bits.div_ceil(window_bits);
        let bucket_count = (1usize << window_bits) - 1;

        let scalar_bigints: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();

        let mut result = G::Group::zero();

        for window_idx in (0..num_windows).rev() {
            for _ in 0..window_bits {
                result.double_in_place();
            }

            let bit_offset = window_idx * window_bits;
            let mut buckets = vec![G::Group::zero(); bucket_count + 1];

            for (base_idx, base) in fixed_bases.iter().enumerate() {
                if base.is_zero() {
                    continue;
                }

                let bigint = &scalar_bigints[base_idx];
                let digit = extract_window_u64(bigint.as_ref(), bit_offset, window_bits);

                if digit != 0 {
                    buckets[digit] += base.into_group();
                }
            }

            let mut running = G::Group::zero();
            let mut window_sum = G::Group::zero();

            for digit in (1..=bucket_count).rev() {
                running += buckets[digit];
                window_sum += running;
            }

            result += window_sum;
        }

        result
    }
}

/// Additional methods for the SRS structure.
impl<G: CommitmentCurve> SRS<G> {
    /// Verify a batch of polynomial commitment opening proofs.
    /// Return `true` if verification succeeds, `false` otherwise.
    pub fn verify<EFqSponge, RNG, const FULL_ROUNDS: usize>(
        &self,
        group_map: &G::Map,
        batch: &mut [BatchEvaluationProof<
            G,
            EFqSponge,
            OpeningProof<G, FULL_ROUNDS>,
            FULL_ROUNDS,
        >],
        rng: &mut RNG,
    ) -> bool
    where
        EFqSponge: FqSponge<G::BaseField, G, G::ScalarField, FULL_ROUNDS>,
        RNG: RngCore + CryptoRng,
        G::BaseField: PrimeField,
    {
        // Verifier checks for all i:
        // c_i Q_i + delta_i = z1_i (G_i + b_i U_i) + z2_i H
        //
        // Sampled at random evalscale, it suffices to check:
        // 0 == sum_i evalscale^i (c_i Q_i + delta_i - (z1_i (G_i + b_i U_i) + z2_i H))
        //
        // G_i is a multiexp on self.g, so we batch across proofs.
        // We also verify that the sg component equals the polynomial commitment to s.

        let nonzero_length = self.g.len();
        let max_rounds = math::ceil_log2(nonzero_length);
        let padded_length = 1 << max_rounds;
        let (_, endo_r) = endos::<G>();

        let padding = padded_length - nonzero_length;

        // Fixed-base points: H followed by G[0..n] followed by zero padding
        let mut points = vec![self.h];
        points.extend(self.g.clone());
        points.extend(vec![G::zero(); padding]);

        // Fixed-base scalars — same length as points
        let mut scalars = vec![G::ScalarField::zero(); padded_length + 1];
        assert_eq!(scalars.len(), points.len());

        // Random combiners sampled once for the whole batch
        let rand_base = G::ScalarField::rand(rng);
        let sg_rand_base = G::ScalarField::rand(rng);

        let mut rand_base_i = G::ScalarField::one();
        let mut sg_rand_base_i = G::ScalarField::one();

        println!("cycle-tracker-start: ipa_build_vectors");

        for BatchEvaluationProof {
            sponge,
            evaluation_points,
            polyscale,
            evalscale,
            evaluations,
            opening,
            combined_inner_product,
        } in batch.iter_mut()
        {
            sponge.absorb_fr(&[shift_scalar::<G>(*combined_inner_product)]);

            // Derive the base point U from the sponge challenge
            let u_base: G = {
                let t = sponge.challenge_fq();
                let (x, y) = group_map.to_group(t);
                G::of_coordinates(x, y)
            };

            let Challenges { chal, chal_inv } = opening.challenges::<EFqSponge>(&endo_r, sponge);

            sponge.absorb_g(&[opening.delta]);
            let c = ScalarChallenge(sponge.challenge()).to_field(&endo_r);

            // b0 = < s, sum_i evalscale^i pows(evaluation_points[i]) >
            //    = sum_i evalscale^i b_poly(chal, evaluation_points[i])
            let b0 = {
                let mut scale = G::ScalarField::one();
                let mut res = G::ScalarField::zero();
                for &e in evaluation_points.iter() {
                    let term = b_poly(&chal, e);
                    res += &(scale * term);
                    scale *= *evalscale;
                }
                res
            };

            // s = b_poly_coefficients(chal) — the vector such that <s, G> = opening.sg
            let s = b_poly_coefficients(&chal);

            let neg_rand_base_i = -rand_base_i;

            // TERM: -rand_base_i * z1 * opening.sg
            //       -sg_rand_base_i * opening.sg   (binding check part 1)
            points.push(opening.sg);
            scalars.push(neg_rand_base_i * opening.z1 - sg_rand_base_i);

            // TERM: sg_rand_base_i * <s, self.g>   (binding check part 2)
            // Together with the term above, enforces opening.sg == <s, G>
            // in the final zero-check.
            {
                #[cfg(not(target_os = "zkvm"))]
                let terms: Vec<_> = s.par_iter().map(|s| sg_rand_base_i * s).collect();

                // On SP1 — sequential iteration, par_iter has no benefit
                #[cfg(target_os = "zkvm")]
                let terms: Vec<_> = s.iter().map(|s| sg_rand_base_i * s).collect();

                for (i, term) in terms.iter().enumerate() {
                    scalars[i + 1] += term;
                }
            }

            // TERM: -rand_base_i * z2 * H
            scalars[0] -= &(rand_base_i * opening.z2);

            // TERM: -rand_base_i * z1 * b0 * U
            points.push(u_base);
            scalars.push(neg_rand_base_i * (opening.z1 * b0));

            // TERM: rand_base_i * c_i * (sum_j chal_inv[j] L[j] + chal[j] R[j] + P')
            let rand_base_i_c_i = c * rand_base_i;
            for ((l, r), (u_inv, u)) in opening.lr.iter().zip(chal_inv.iter().zip(chal.iter())) {
                points.push(*l);
                scalars.push(rand_base_i_c_i * u_inv);
                points.push(*r);
                scalars.push(rand_base_i_c_i * u);
            }

            // TERM: sum_j evalscale^j (sum_i polyscale^i f_i)(elm_j)
            combine_commitments(
                evaluations,
                &mut scalars,
                &mut points,
                *polyscale,
                rand_base_i_c_i,
            );

            // TERM: rand_base_i * c_i * combined_inner_product * U
            points.push(u_base);
            scalars.push(rand_base_i_c_i * *combined_inner_product);

            // TERM: rand_base_i * delta
            points.push(opening.delta);
            scalars.push(rand_base_i);

            rand_base_i *= &rand_base;
            sg_rand_base_i *= &sg_rand_base;
        }

        println!("cycle-tracker-end: ipa_build_vectors");

        // ------------------------------------------------------------------
        // Final MSM — result must be zero for the proof to be valid
        // ------------------------------------------------------------------

        let scalars_bigint: Vec<_> = scalars.iter().map(|x| x.into_bigint()).collect();

        println!("cycle-tracker-start: ipa_fixed_msm");

        #[cfg(not(target_os = "zkvm"))]
        let msm_res = {
            // Non-SP1: parallel chunked MSM — optimal for large SRS on multi-core
            let chunk_size = points.len() / 2;
            points
                .into_par_iter()
                .chunks(chunk_size)
                .zip(scalars_bigint.into_par_iter().chunks(chunk_size))
                .map(|(bases, coeffs)| G::Group::msm_bigint(&bases, &coeffs))
                .reduce(G::Group::zero, |mut l, r| {
                    l += r;
                    l
                })
        };

        #[cfg(target_os = "zkvm")]
        let msm_res = {
            // SP1: single sequential MSM — no parallelism overhead on RISC-V
            G::Group::msm_bigint(&points, &scalars_bigint)
        };

        println!("cycle-tracker-end: ipa_fixed_msm");

        msm_res == G::Group::zero()
    }

    /// Create a trusted-setup SRS instance for circuits with
    /// number of rows up to `depth`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a trusted setup and the toxic
    /// waste is passed as a parameter.
    pub unsafe fn create_trusted_setup(x: G::ScalarField, depth: usize) -> Self {
        let m = G::Map::setup();

        let mut x_pow = G::ScalarField::one();
        let g: Vec<_> = (0..depth)
            .map(|_| {
                let res = G::generator().mul(x_pow);
                x_pow *= x;
                res.into_affine()
            })
            .collect();

        // Compute a blinder.
        let h = {
            let mut h = Blake2b512::new();
            h.update("srs_misc".as_bytes());
            // This is kept for retrocompatibility with a previous version.
            h.update(0_u32.to_be_bytes());
            point_of_random_bytes(&m, &h.finalize())
        };

        Self {
            g,
            h,
            lagrange_bases: HashMapCache::new(),
        }
    }
}

impl<G: CommitmentCurve> SRS<G>
where
    <G as CommitmentCurve>::Map: Sync,
    G::BaseField: PrimeField,
{
    /// Create an SRS instance for circuits with number of rows up
    /// to `depth`.
    pub fn create_parallel(depth: usize) -> Self {
        let m = G::Map::setup();

        let g: Vec<_> = (0..depth)
            .into_par_iter()
            .map(|i| {
                let mut h = Blake2b512::new();
                h.update((i as u32).to_be_bytes());
                point_of_random_bytes(&m, &h.finalize())
            })
            .collect();

        // Compute a blinder.
        let h = {
            let mut h = Blake2b512::new();
            h.update("srs_misc".as_bytes());
            // This is kept for retrocompatibility with a previous version.
            h.update(0_u32.to_be_bytes());
            point_of_random_bytes(&m, &h.finalize())
        };

        Self {
            g,
            h,
            lagrange_bases: HashMapCache::new(),
        }
    }
}

impl<G> SRSTrait<G> for SRS<G>
where
    G: CommitmentCurve,
{
    /// The maximum polynomial degree that can be committed to.
    fn max_poly_size(&self) -> usize {
        self.g.len()
    }

    fn blinding_commitment(&self) -> G {
        self.h
    }

    /// Turn a non-hiding polynomial commitment into a hiding polynomial
    /// commitment.
    fn mask(
        &self,
        comm: PolyComm<G>,
        rng: &mut (impl RngCore + CryptoRng),
    ) -> BlindedCommitment<G> {
        let blinders = comm.map(|_| G::ScalarField::rand(rng));
        self.mask_custom(comm, &blinders).unwrap()
    }

    fn mask_custom(
        &self,
        com: PolyComm<G>,
        blinders: &PolyComm<G::ScalarField>,
    ) -> Result<BlindedCommitment<G>, CommitmentError> {
        let commitment = com
            .zip(blinders)
            .ok_or_else(|| CommitmentError::BlindersDontMatch(blinders.len(), com.len()))?
            .map(|(g, b)| {
                let mut g_masked = self.h.mul(b);
                g_masked.add_assign(&g);
                g_masked.into_affine()
            });
        Ok(BlindedCommitment {
            commitment,
            blinders: blinders.clone(),
        })
    }

    fn commit_non_hiding(
        &self,
        plnm: &DensePolynomial<G::ScalarField>,
        num_chunks: usize,
    ) -> PolyComm<G> {
        let is_zero = plnm.is_zero();

        // Chunk while committing.
        let mut chunks: Vec<_> = if is_zero {
            vec![G::zero()]
        } else if plnm.len() < self.g.len() {
            vec![G::Group::msm(&self.g[..plnm.len()], &plnm.coeffs)
                .unwrap()
                .into_affine()]
        } else if plnm.len() == self.g.len() {
            // When processing a single chunk, it is faster to parallelize
            // vertically in 2 threads.
            let n = self.g.len();
            let (r1, r2) = rayon::join(
                || G::Group::msm(&self.g[..n / 2], &plnm.coeffs[..n / 2]).unwrap(),
                || G::Group::msm(&self.g[n / 2..n], &plnm.coeffs[n / 2..n]).unwrap(),
            );

            vec![(r1 + r2).into_affine()]
        } else {
            // Otherwise it is better to parallelize horizontally along chunks.
            plnm.into_par_iter()
                .chunks(self.g.len())
                .map(|chunk| {
                    let chunk_coeffs = chunk
                        .into_iter()
                        .map(|c| c.into_bigint())
                        .collect::<Vec<_>>();
                    let chunk_res = G::Group::msm_bigint(&self.g, &chunk_coeffs);
                    chunk_res.into_affine()
                })
                .collect()
        };

        for _ in chunks.len()..num_chunks {
            chunks.push(G::zero());
        }

        PolyComm::<G>::new(chunks)
    }

    fn commit(
        &self,
        plnm: &DensePolynomial<G::ScalarField>,
        num_chunks: usize,
        rng: &mut (impl RngCore + CryptoRng),
    ) -> BlindedCommitment<G> {
        self.mask(self.commit_non_hiding(plnm, num_chunks), rng)
    }

    fn commit_custom(
        &self,
        plnm: &DensePolynomial<G::ScalarField>,
        num_chunks: usize,
        blinders: &PolyComm<G::ScalarField>,
    ) -> Result<BlindedCommitment<G>, CommitmentError> {
        self.mask_custom(self.commit_non_hiding(plnm, num_chunks), blinders)
    }

    fn commit_evaluations_non_hiding(
        &self,
        domain: D<G::ScalarField>,
        plnm: &Evaluations<G::ScalarField, D<G::ScalarField>>,
    ) -> PolyComm<G> {
        let basis = self.get_lagrange_basis(domain);
        let commit_evaluations = |evals: &Vec<G::ScalarField>, basis: &Vec<PolyComm<G>>| {
            PolyComm::<G>::multi_scalar_mul(&basis.iter().collect::<Vec<_>>()[..], &evals[..])
        };
        match domain.size.cmp(&plnm.domain().size) {
            std::cmp::Ordering::Less => {
                let s = (plnm.domain().size / domain.size) as usize;
                let v: Vec<_> = (0..(domain.size())).map(|i| plnm.evals[s * i]).collect();
                commit_evaluations(&v, basis)
            }
            std::cmp::Ordering::Equal => commit_evaluations(&plnm.evals, basis),
            std::cmp::Ordering::Greater => {
                panic!(
                    "desired commitment domain size ({}) greater than evaluations' domain size ({}):",
                    domain.size,
                    plnm.domain().size
                )
            }
        }
    }

    fn commit_evaluations(
        &self,
        domain: D<G::ScalarField>,
        plnm: &Evaluations<G::ScalarField, D<G::ScalarField>>,
        rng: &mut (impl RngCore + CryptoRng),
    ) -> BlindedCommitment<G> {
        self.mask(self.commit_evaluations_non_hiding(domain, plnm), rng)
    }

    fn commit_evaluations_custom(
        &self,
        domain: D<G::ScalarField>,
        plnm: &Evaluations<G::ScalarField, D<G::ScalarField>>,
        blinders: &PolyComm<G::ScalarField>,
    ) -> Result<BlindedCommitment<G>, CommitmentError> {
        self.mask_custom(self.commit_evaluations_non_hiding(domain, plnm), blinders)
    }

    fn create(depth: usize) -> Self {
        let m = G::Map::setup();

        let g: Vec<_> = (0..depth)
            .map(|i| {
                let mut h = Blake2b512::new();
                h.update((i as u32).to_be_bytes());
                point_of_random_bytes(&m, &h.finalize())
            })
            .collect();

        // Compute a blinder.
        let h = {
            let mut h = Blake2b512::new();
            h.update("srs_misc".as_bytes());
            // This is kept for retrocompatibility with a previous version.
            h.update(0_u32.to_be_bytes());
            point_of_random_bytes(&m, &h.finalize())
        };

        Self {
            g,
            h,
            lagrange_bases: HashMapCache::new(),
        }
    }

    fn get_lagrange_basis_from_domain_size(&self, domain_size: usize) -> &Vec<PolyComm<G>> {
        self.lagrange_bases.get_or_generate(domain_size, || {
            self.lagrange_basis(D::new(domain_size).unwrap())
        })
    }

    fn get_lagrange_basis(&self, domain: D<G::ScalarField>) -> &Vec<PolyComm<G>> {
        self.lagrange_bases
            .get_or_generate(domain.size(), || self.lagrange_basis(domain))
    }

    fn size(&self) -> usize {
        self.g.len()
    }
}

impl<G: CommitmentCurve> SRS<G> {
    #[allow(clippy::type_complexity)]
    #[allow(clippy::many_single_char_names)]
    // A slight modification to the original protocol is done when absorbing
    // the first prover message to improve efficiency in a recursive setting.
    pub fn open<EFqSponge, RNG, D: EvaluationDomain<G::ScalarField>, const FULL_ROUNDS: usize>(
        &self,
        group_map: &G::Map,
        plnms: PolynomialsToCombine<G, D>,
        elm: &[G::ScalarField],
        polyscale: G::ScalarField,
        evalscale: G::ScalarField,
        mut sponge: EFqSponge,
        rng: &mut RNG,
    ) -> OpeningProof<G, FULL_ROUNDS>
    where
        EFqSponge: Clone + FqSponge<G::BaseField, G, G::ScalarField, FULL_ROUNDS>,
        RNG: RngCore + CryptoRng,
        G::BaseField: PrimeField,
        G: EndoCurve,
    {
        let (endo_q, endo_r) = endos::<G>();

        let rounds = math::ceil_log2(self.g.len());
        let padded_length = 1 << rounds;

        // We usually have a power-of-two SRS, so padding is zero in practice.
        let padding = padded_length - self.g.len();
        let mut g = self.g.clone();
        g.extend(vec![G::zero(); padding]);

        // Combine polynomials roughly as:
        // p(X) := Σ_i polyscale^i p_i(X)
        let (p, blinding_factor) = combine_polys::<G, D>(plnms, polyscale, self.g.len());

        // Build the combined evaluation vector.
        let b_init = {
            let mut scale = G::ScalarField::one();
            let mut res: Vec<G::ScalarField> =
                (0..padded_length).map(|_| G::ScalarField::zero()).collect();
            for e in elm {
                for (i, t) in pows(padded_length, *e).iter().enumerate() {
                    res[i] += &(scale * t);
                }
                scale *= &evalscale;
            }
            res
        };

        let combined_inner_product = p
            .coeffs
            .iter()
            .zip(b_init.iter())
            .map(|(a, b)| *a * b)
            .fold(G::ScalarField::zero(), |acc, x| acc + x);

        sponge.absorb_fr(&[shift_scalar::<G>(combined_inner_product)]);

        // Generate another randomization base U.
        let u_base: G = {
            let t = sponge.challenge_fq();
            let (x, y) = group_map.to_group(t);
            G::of_coordinates(x, y)
        };

        let mut a = p.coeffs;
        assert!(padded_length >= a.len());
        a.extend(vec![G::ScalarField::zero(); padded_length - a.len()]);

        let mut b = b_init;

        let mut lr = vec![];
        let mut blinders = vec![];
        let mut chals = vec![];
        let mut chal_invs = vec![];

        // Main IPA folding loop with logarithmic number of rounds.
        for _ in 0..rounds {
            let n = g.len() / 2;
            let (g_lo, g_hi) = (&g[0..n], &g[n..]);
            let (a_lo, a_hi) = (&a[0..n], &a[n..]);
            let (b_lo, b_hi) = (&b[0..n], &b[n..]);

            let rand_l = <G::ScalarField as UniformRand>::rand(rng);
            let rand_r = <G::ScalarField as UniformRand>::rand(rng);

            let l = G::Group::msm_bigint(
                &[g_lo, &[self.h, u_base]].concat(),
                &[a_hi, &[rand_l, inner_prod(a_hi, b_lo)]]
                    .concat()
                    .iter()
                    .map(|x| x.into_bigint())
                    .collect::<Vec<_>>(),
            )
            .into_affine();

            let r = G::Group::msm_bigint(
                &[g_hi, &[self.h, u_base]].concat(),
                &[a_lo, &[rand_r, inner_prod(a_lo, b_hi)]]
                    .concat()
                    .iter()
                    .map(|x| x.into_bigint())
                    .collect::<Vec<_>>(),
            )
            .into_affine();

            lr.push((l, r));
            blinders.push((rand_l, rand_r));

            sponge.absorb_g(&[l]);
            sponge.absorb_g(&[r]);

            let u_pre = squeeze_prechallenge(&mut sponge);
            let u = u_pre.to_field(&endo_r);
            let u_inv = u.inverse().unwrap();

            chals.push(u);
            chal_invs.push(u_inv);

            a = a_hi
                .par_iter()
                .zip(a_lo)
                .map(|(&hi, &lo)| {
                    let mut res = hi;
                    res *= u_inv;
                    res += &lo;
                    res
                })
                .collect();

            b = b_lo
                .par_iter()
                .zip(b_hi)
                .map(|(&lo, &hi)| {
                    let mut res = hi;
                    res *= u;
                    res += &lo;
                    res
                })
                .collect();

            g = G::combine_one_endo(endo_r, endo_q, g_lo, g_hi, u_pre);
        }

        assert!(
            g.len() == 1 && a.len() == 1 && b.len() == 1,
            "IPA commitment folding must produce single elements after log rounds"
        );

        let a0 = a[0];
        let b0 = b[0];
        let g0 = g[0];

        let r_prime = blinders
            .iter()
            .zip(chals.iter().zip(chal_invs.iter()))
            .map(|((rand_l, rand_r), (u, u_inv))| ((*rand_l) * u_inv) + (*rand_r * u))
            .fold(blinding_factor, |acc, x| acc + x);

        let d = <G::ScalarField as UniformRand>::rand(rng);
        let r_delta = <G::ScalarField as UniformRand>::rand(rng);

        let delta = ((g0.into_group() + (u_base.mul(b0))).into_affine().mul(d)
            + self.h.mul(r_delta))
        .into_affine();

        sponge.absorb_g(&[delta]);
        let c = ScalarChallenge(sponge.challenge()).to_field(&endo_r);

        let z1 = a0 * c + d;
        let z2 = r_prime * c + r_delta;

        OpeningProof {
            delta,
            lr,
            z1,
            z2,
            sg: g0,
        }
    }

    fn lagrange_basis(&self, domain: D<G::ScalarField>) -> Vec<PolyComm<G>> {
        let n = domain.size();

        let srs_size = self.g.len();
        let num_elems = n.div_ceil(srs_size);
        let mut chunks = Vec::with_capacity(num_elems);

        for i in 0..num_elems {
            let mut lg: Vec<<G as AffineRepr>::Group> = vec![<G as AffineRepr>::Group::zero(); n];
            let start_offset = i * srs_size;
            let num_terms = min((i + 1) * srs_size, n) - start_offset;
            for j in 0..num_terms {
                lg[start_offset + j] = self.g[j].into_group()
            }
            domain.ifft_in_place(&mut lg);
            chunks.push(<G as AffineRepr>::Group::normalize_batch(lg.as_mut_slice()));
        }

        (0..n)
            .map(|i| PolyComm {
                chunks: chunks.iter().map(|v| v[i]).collect(),
            })
            .collect()
    }
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(bound = "G: ark_serialize::CanonicalDeserialize + ark_serialize::CanonicalSerialize")]
pub struct OpeningProof<G: AffineRepr, const FULL_ROUNDS: usize> {
    /// Vector of rounds of L & R commitments.
    #[serde_as(as = "Vec<(o1_utils::serialization::SerdeAs, o1_utils::serialization::SerdeAs)>")]
    pub lr: Vec<(G, G)>,
    #[serde_as(as = "o1_utils::serialization::SerdeAs")]
    pub delta: G,
    #[serde_as(as = "o1_utils::serialization::SerdeAs")]
    pub z1: G::ScalarField,
    #[serde_as(as = "o1_utils::serialization::SerdeAs")]
    pub z2: G::ScalarField,
    /// A final folded commitment base.
    #[serde_as(as = "o1_utils::serialization::SerdeAs")]
    pub sg: G,
}

impl<
        BaseField: PrimeField,
        G: AffineRepr<BaseField = BaseField> + CommitmentCurve + EndoCurve,
        const FULL_ROUNDS: usize,
    > crate::OpenProof<G, FULL_ROUNDS> for OpeningProof<G, FULL_ROUNDS>
{
    type SRS = SRS<G>;

    fn open<EFqSponge, RNG, D: EvaluationDomain<<G as AffineRepr>::ScalarField>>(
        srs: &Self::SRS,
        group_map: &<G as CommitmentCurve>::Map,
        plnms: PolynomialsToCombine<G, D>,
        elm: &[<G as AffineRepr>::ScalarField],
        polyscale: <G as AffineRepr>::ScalarField,
        evalscale: <G as AffineRepr>::ScalarField,
        sponge: EFqSponge,
        rng: &mut RNG,
    ) -> Self
    where
        EFqSponge: Clone
            + FqSponge<<G as AffineRepr>::BaseField, G, <G as AffineRepr>::ScalarField, FULL_ROUNDS>,
        RNG: RngCore + CryptoRng,
    {
        srs.open(group_map, plnms, elm, polyscale, evalscale, sponge, rng)
    }

    fn verify<EFqSponge, RNG>(
        srs: &Self::SRS,
        group_map: &G::Map,
        batch: &mut [BatchEvaluationProof<G, EFqSponge, Self, FULL_ROUNDS>],
        rng: &mut RNG,
    ) -> bool
    where
        EFqSponge:
            FqSponge<<G as AffineRepr>::BaseField, G, <G as AffineRepr>::ScalarField, FULL_ROUNDS>,
        RNG: RngCore + CryptoRng,
    {
        srs.verify(group_map, batch, rng)
    }
}

/// Commitment round challenges (endo mapped) and their inverses.
pub struct Challenges<F> {
    pub chal: Vec<F>,
    pub chal_inv: Vec<F>,
}

impl<G: AffineRepr, const FULL_ROUNDS: usize> OpeningProof<G, FULL_ROUNDS> {
    /// Compute a log-sized vector of scalar challenges for
    /// recombining elements inside the IPA.
    pub fn prechallenges<EFqSponge: FqSponge<G::BaseField, G, G::ScalarField, FULL_ROUNDS>>(
        &self,
        sponge: &mut EFqSponge,
    ) -> Vec<ScalarChallenge<G::ScalarField>> {
        let _t = sponge.challenge_fq();
        self.lr
            .iter()
            .map(|(l, r)| {
                sponge.absorb_g(&[*l]);
                sponge.absorb_g(&[*r]);
                squeeze_prechallenge(sponge)
            })
            .collect()
    }

    /// Same as `prechallenges`, but map scalar challenges using the provided
    /// endomorphism and compute their inverses.
    pub fn challenges<EFqSponge: FqSponge<G::BaseField, G, G::ScalarField, FULL_ROUNDS>>(
        &self,
        endo_r: &G::ScalarField,
        sponge: &mut EFqSponge,
    ) -> Challenges<G::ScalarField> {
        let chal: Vec<_> = self
            .lr
            .iter()
            .map(|(l, r)| {
                sponge.absorb_g(&[*l]);
                sponge.absorb_g(&[*r]);
                squeeze_challenge(endo_r, sponge)
            })
            .collect();

        let chal_inv = {
            let mut cs = chal.clone();
            ark_ff::batch_inversion(&mut cs);
            cs
        };

        Challenges { chal, chal_inv }
    }
}

#[cfg(feature = "ocaml_types")]
#[allow(non_local_definitions)]
pub mod caml {
    use super::OpeningProof;
    use ark_ec::AffineRepr;
    use ocaml;

    #[derive(ocaml::IntoValue, ocaml::FromValue, ocaml_gen::Struct)]
    pub struct CamlOpeningProof<G, F> {
        /// Vector of rounds of L & R commitments.
        pub lr: Vec<(G, G)>,
        pub delta: G,
        pub z1: F,
        pub z2: F,
        pub sg: G,
    }

    impl<G, CamlF, CamlG, const FULL_ROUNDS: usize> From<OpeningProof<G, FULL_ROUNDS>>
        for CamlOpeningProof<CamlG, CamlF>
    where
        G: AffineRepr,
        CamlG: From<G>,
        CamlF: From<G::ScalarField>,
    {
        fn from(opening_proof: OpeningProof<G, FULL_ROUNDS>) -> Self {
            Self {
                lr: opening_proof
                    .lr
                    .into_iter()
                    .map(|(g1, g2)| (CamlG::from(g1), CamlG::from(g2)))
                    .collect(),
                delta: CamlG::from(opening_proof.delta),
                z1: opening_proof.z1.into(),
                z2: opening_proof.z2.into(),
                sg: CamlG::from(opening_proof.sg),
            }
        }
    }

    impl<G, CamlF, CamlG, const FULL_ROUNDS: usize> From<CamlOpeningProof<CamlG, CamlF>>
        for OpeningProof<G, FULL_ROUNDS>
    where
        G: AffineRepr,
        CamlG: Into<G>,
        CamlF: Into<G::ScalarField>,
    {
        fn from(caml: CamlOpeningProof<CamlG, CamlF>) -> Self {
            Self {
                lr: caml
                    .lr
                    .into_iter()
                    .map(|(g1, g2)| (g1.into(), g2.into()))
                    .collect(),
                delta: caml.delta.into(),
                z1: caml.z1.into(),
                z2: caml.z2.into(),
                sg: caml.sg.into(),
            }
        }
    }
}
