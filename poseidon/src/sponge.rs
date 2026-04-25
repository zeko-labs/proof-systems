extern crate alloc;

use crate::{
    constants::{PlonkSpongeConstantsKimchi, SpongeConstants},
    poseidon::{ArithmeticSponge, ArithmeticSpongeParams, Sponge},
};
use alloc::{vec, vec::Vec};
use ark_ec::models::short_weierstrass::{Affine, SWCurveConfig};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Abstracts a sponge operating on a base field `Fq` of the curve
/// `G`. The parameter `Fr` models the scalar field of the curve.
pub trait FqSponge<Fq: Field, G, Fr, const FULL_ROUNDS: usize> {
    /// Creates a new sponge.
    fn new(p: &'static ArithmeticSpongeParams<Fq, FULL_ROUNDS>) -> Self;

    /// Absorbs base field elements.
    fn absorb_fq(&mut self, x: &[Fq]);

    /// Absorbs curve points.
    /// The point at infinity is encoded as `(0, 0)`.
    fn absorb_g(&mut self, g: &[G]);

    /// Absorbs scalar field elements by converting them to the base field first.
    fn absorb_fr(&mut self, x: &[Fr]);

    /// Squeezes a base field challenge.
    fn challenge_fq(&mut self) -> Fq;

    /// Squeezes a scalar field challenge.
    fn challenge(&mut self) -> Fr;

    /// Returns a base field digest.
    fn digest_fq(self) -> Fq;

    /// Returns a scalar field digest.
    fn digest(self) -> Fr;
}

pub const CHALLENGE_LENGTH_IN_LIMBS: usize = 2;
const HIGH_ENTROPY_LIMBS: usize = 2;

/// A challenge used as a scalar on a group element in the verifier.
#[derive(Clone, Debug)]
pub struct ScalarChallenge<F>(pub F);

pub fn endo_coefficient<F: PrimeField>() -> F {
    let p_minus_1_over_3 = (F::zero() - F::one()) / F::from(3u64);
    F::GENERATOR.pow(p_minus_1_over_3.into_bigint().as_ref())
}

#[inline(always)]
fn get_bit(limbs_lsb: &[u64], i: u64) -> u64 {
    let limb = i / 64;
    let j = i % 64;
    (limbs_lsb[limb as usize] >> j) & 1
}

impl<F: PrimeField> ScalarChallenge<F> {
    pub fn to_field_with_length(&self, length_in_bits: usize, endo_coeff: &F) -> F {
        let rep = self.0.into_bigint();
        let r = rep.as_ref();

        let mut a: F = 2_u64.into();
        let mut b: F = 2_u64.into();

        let one = F::one();
        let neg_one = -one;

        for i in (0..(length_in_bits as u64 / 2)).rev() {
            a.double_in_place();
            b.double_in_place();

            let r_2i = get_bit(r, 2 * i);
            let s = if r_2i == 0 { &neg_one } else { &one };

            if get_bit(r, 2 * i + 1) == 0 {
                b += s;
            } else {
                a += s;
            }
        }

        a * endo_coeff + b
    }

    pub fn to_field(&self, endo_coeff: &F) -> F {
        let length_in_bits = 64 * CHALLENGE_LENGTH_IN_LIMBS;
        self.to_field_with_length(length_in_bits, endo_coeff)
    }
}

#[derive(Clone)]
pub struct DefaultFqSponge<P: SWCurveConfig, SC: SpongeConstants, const FULL_ROUNDS: usize>
where
    P::BaseField: CanonicalSerialize + CanonicalDeserialize,
{
    pub sponge: ArithmeticSponge<P::BaseField, SC, FULL_ROUNDS>,
    pub last_squeezed: Vec<u64>,
}

pub struct DefaultFrSponge<
    Fr: Field + CanonicalSerialize + CanonicalDeserialize,
    SC: SpongeConstants,
    const FULL_ROUNDS: usize,
> {
    pub sponge: ArithmeticSponge<Fr, SC, FULL_ROUNDS>,
    pub last_squeezed: Vec<u64>,
}

impl<const FULL_ROUNDS: usize, Fr> From<&'static ArithmeticSpongeParams<Fr, FULL_ROUNDS>>
    for DefaultFrSponge<Fr, PlonkSpongeConstantsKimchi, FULL_ROUNDS>
where
    Fr: PrimeField + CanonicalSerialize + CanonicalDeserialize,
{
    fn from(p: &'static ArithmeticSpongeParams<Fr, FULL_ROUNDS>) -> Self {
        DefaultFrSponge {
            sponge: ArithmeticSponge::new(p),
            last_squeezed: vec![],
        }
    }
}

#[inline(always)]
fn pack<B: BigInteger>(limbs_lsb: &[u64]) -> B {
    let mut res: B = 0u64.into();
    for &x in limbs_lsb.iter().rev() {
        res <<= 64;
        res.add_with_carry(&x.into());
    }
    res
}

#[inline(always)]
fn take_first_limbs(buf: &mut Vec<u64>, num_limbs: usize) -> Vec<u64> {
    debug_assert!(buf.len() >= num_limbs);

    let mut out = Vec::with_capacity(num_limbs);
    out.extend_from_slice(&buf[..num_limbs]);

    let remaining = buf.len() - num_limbs;
    for i in 0..remaining {
        buf[i] = buf[num_limbs + i];
    }
    buf.truncate(remaining);

    out
}

impl<
        Fr: PrimeField + CanonicalSerialize + CanonicalDeserialize,
        SC: SpongeConstants,
        const FULL_ROUNDS: usize,
    > DefaultFrSponge<Fr, SC, FULL_ROUNDS>
{
    #[inline(always)]
    fn refill_limbs(&mut self) {
        let x = self.sponge.squeeze().into_bigint();
        self.last_squeezed
            .extend_from_slice(&x.as_ref()[0..HIGH_ENTROPY_LIMBS]);
    }

    pub fn squeeze(&mut self, num_limbs: usize) -> Fr {
        while self.last_squeezed.len() < num_limbs {
            self.refill_limbs();
        }

        Fr::from(pack::<Fr::BigInt>(&take_first_limbs(
            &mut self.last_squeezed,
            num_limbs,
        )))
    }
}

impl<P: SWCurveConfig, SC: SpongeConstants, const FULL_ROUNDS: usize>
    DefaultFqSponge<P, SC, FULL_ROUNDS>
where
    P::BaseField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    <P::BaseField as PrimeField>::BigInt: Into<<P::ScalarField as PrimeField>::BigInt>,
{
    #[inline(always)]
    fn refill_limbs(&mut self) {
        let x = self.sponge.squeeze().into_bigint();
        self.last_squeezed
            .extend_from_slice(&x.as_ref()[0..HIGH_ENTROPY_LIMBS]);
    }

    pub fn squeeze_limbs(&mut self, num_limbs: usize) -> Vec<u64> {
        while self.last_squeezed.len() < num_limbs {
            self.refill_limbs();
        }

        take_first_limbs(&mut self.last_squeezed, num_limbs)
    }

    pub fn squeeze_field(&mut self) -> P::BaseField {
        self.last_squeezed.clear();
        self.sponge.squeeze()
    }

    pub fn squeeze(&mut self, num_limbs: usize) -> P::ScalarField {
        P::ScalarField::from_bigint(pack(&self.squeeze_limbs(num_limbs)))
            .expect("internal representation was not a valid field element")
    }
}

impl<P: SWCurveConfig, SC: SpongeConstants, const FULL_ROUNDS: usize>
    FqSponge<P::BaseField, Affine<P>, P::ScalarField, FULL_ROUNDS>
    for DefaultFqSponge<P, SC, FULL_ROUNDS>
where
    P::BaseField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    <P::BaseField as PrimeField>::BigInt: Into<<P::ScalarField as PrimeField>::BigInt>,
{
    fn new(params: &'static ArithmeticSpongeParams<P::BaseField, FULL_ROUNDS>) -> Self {
        let sponge = ArithmeticSponge::new(params);
        DefaultFqSponge {
            sponge,
            last_squeezed: vec![],
        }
    }

    fn absorb_g(&mut self, g: &[Affine<P>]) {
        self.last_squeezed.clear();

        let mut buf = Vec::with_capacity(2 * g.len());
        for point in g.iter() {
            if point.infinity {
                // Absorb a fake point (0, 0).
                let zero = P::BaseField::zero();
                buf.push(zero);
                buf.push(zero);
            } else {
                buf.push(point.x);
                buf.push(point.y);
            }
        }

        self.sponge.absorb(&buf);
    }

    fn absorb_fq(&mut self, x: &[P::BaseField]) {
        self.last_squeezed.clear();
        self.sponge.absorb(x);
    }

    fn absorb_fr(&mut self, x: &[P::ScalarField]) {
        self.last_squeezed.clear();

        if <P::ScalarField as PrimeField>::MODULUS < <P::BaseField as PrimeField>::MODULUS.into() {
            let mut buf = Vec::with_capacity(x.len());

            for scalar in x.iter() {
                let bits = scalar.into_bigint().to_bits_le();
                let fe = P::BaseField::from_bigint(
                    <P::BaseField as PrimeField>::BigInt::from_bits_le(&bits),
                )
                .expect("padding code has a bug");
                buf.push(fe);
            }

            self.sponge.absorb(&buf);
        } else {
            let mut buf = Vec::with_capacity(2 * x.len());

            for scalar in x.iter() {
                let bits = scalar.into_bigint().to_bits_le();

                let low_bit = if bits[0] {
                    P::BaseField::one()
                } else {
                    P::BaseField::zero()
                };

                let high_bits = P::BaseField::from_bigint(
                    <P::BaseField as PrimeField>::BigInt::from_bits_le(&bits[1..bits.len()]),
                )
                .expect("padding code has a bug");

                buf.push(high_bits);
                buf.push(low_bit);
            }

            self.sponge.absorb(&buf);
        }
    }

    fn digest(mut self) -> P::ScalarField {
        let x: <P::BaseField as PrimeField>::BigInt = self.squeeze_field().into_bigint();

        // Returns zero for values that are too large.
        // This means that there is a bias for the value zero in one of the curves.
        // Since log2(q - p) is much smaller than log2(q), the attack remains negligible.
        P::ScalarField::from_bigint(x.into()).unwrap_or_else(P::ScalarField::zero)
    }

    fn digest_fq(mut self) -> P::BaseField {
        self.squeeze_field()
    }

    fn challenge(&mut self) -> P::ScalarField {
        self.squeeze(CHALLENGE_LENGTH_IN_LIMBS)
    }

    fn challenge_fq(&mut self) -> P::BaseField {
        self.squeeze_field()
    }
}

#[cfg(feature = "ocaml_types")]
#[allow(non_local_definitions)]
pub mod caml {
    use super::*;

    extern crate alloc;
    use alloc::{
        format,
        string::{String, ToString},
    };

    /// ScalarChallenge<F> <-> CamlScalarChallenge<CamlF>
    #[derive(Debug, Clone, ocaml::IntoValue, ocaml::FromValue, ocaml_gen::Struct)]
    pub struct CamlScalarChallenge<CamlF>(pub CamlF);

    impl<F, CamlF> From<ScalarChallenge<F>> for CamlScalarChallenge<CamlF>
    where
        CamlF: From<F>,
    {
        fn from(sc: ScalarChallenge<F>) -> Self {
            Self(sc.0.into())
        }
    }

    impl<F, CamlF> From<CamlScalarChallenge<CamlF>> for ScalarChallenge<F>
    where
        CamlF: Into<F>,
    {
        fn from(caml_sc: CamlScalarChallenge<CamlF>) -> Self {
            Self(caml_sc.0.into())
        }
    }
}
