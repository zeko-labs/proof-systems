mod fp;
mod params;
mod poseidon;

use alloc::vec::Vec;
use core::marker::PhantomData;

use ark_ec::models::short_weierstrass::{Affine, SWCurveConfig};
use ark_ff::{BigInteger, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use fp::Fp as Sp1Fp;
use poseidon::Sponge as Sp1Sponge;

use crate::{
    poseidon::ArithmeticSpongeParams,
    sponge::{FqSponge, ScalarChallenge, CHALLENGE_LENGTH_IN_LIMBS},
};

// -----------------------------------------------------------------------------
// ark <-> SP1 field conversions
// -----------------------------------------------------------------------------

#[inline(always)]
fn ark_to_sp1<F: PrimeField>(x: F) -> Sp1Fp {
    let limbs = x.into_bigint();
    let limbs = limbs.as_ref();

    Sp1Fp::from_le_limbs([limbs[0], limbs[1], limbs[2], limbs[3]])
}

#[inline(always)]
fn sp1_to_ark<F: PrimeField>(x: Sp1Fp) -> F {
    let limbs = x.to_le_limbs();

    let mut bigint = F::BigInt::default();
    bigint.as_mut().copy_from_slice(&limbs);

    F::from_bigint(bigint).expect("SP1 field element is not valid in ark field")
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

    let out = buf[..num_limbs].to_vec();
    let remaining = buf[num_limbs..].to_vec();

    *buf = remaining;

    out
}

// -----------------------------------------------------------------------------
// Fq sponge wrapper
// -----------------------------------------------------------------------------

#[derive(Clone)]
pub struct Sp1FqSponge<P: SWCurveConfig, SC = (), const FULL_ROUNDS: usize = 55> {
    pub sponge: Sp1Sponge,
    pub last_squeezed: Vec<u64>,
    _phantom: PhantomData<(P, SC)>,
}

impl<P, SC, const FULL_ROUNDS: usize> Sp1FqSponge<P, SC, FULL_ROUNDS>
where
    P: SWCurveConfig,
    P::BaseField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    P::ScalarField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    <P::BaseField as PrimeField>::BigInt: Into<<P::ScalarField as PrimeField>::BigInt>,
{
    #[inline(always)]
    fn refill_limbs(&mut self) {
        let x: P::BaseField = sp1_to_ark(self.sponge.squeeze());
        let bigint = x.into_bigint();

        self.last_squeezed
            .extend_from_slice(&bigint.as_ref()[0..CHALLENGE_LENGTH_IN_LIMBS]);
    }

    #[inline(always)]
    pub fn squeeze_limbs(&mut self, num_limbs: usize) -> Vec<u64> {
        while self.last_squeezed.len() < num_limbs {
            self.refill_limbs();
        }

        take_first_limbs(&mut self.last_squeezed, num_limbs)
    }

    #[inline(always)]
    pub fn squeeze_field(&mut self) -> P::BaseField {
        self.last_squeezed.clear();
        sp1_to_ark(self.sponge.squeeze())
    }

    #[inline(always)]
    pub fn squeeze(&mut self, num_limbs: usize) -> P::ScalarField {
        P::ScalarField::from_bigint(pack(&self.squeeze_limbs(num_limbs)))
            .expect("squeezed scalar is not a valid scalar field element")
    }
}

impl<P, SC, const FULL_ROUNDS: usize>
    FqSponge<P::BaseField, Affine<P>, P::ScalarField, FULL_ROUNDS>
    for Sp1FqSponge<P, SC, FULL_ROUNDS>
where
    P: SWCurveConfig,
    P::BaseField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    P::ScalarField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    <P::BaseField as PrimeField>::BigInt: Into<<P::ScalarField as PrimeField>::BigInt>,
{
    fn new(_params: &'static ArithmeticSpongeParams<P::BaseField, FULL_ROUNDS>) -> Self {
        Self {
            sponge: Sp1Sponge::new(),
            last_squeezed: Vec::new(),
            _phantom: PhantomData,
        }
    }

    fn absorb_fq(&mut self, x: &[P::BaseField]) {
        self.last_squeezed.clear();

        let inputs: Vec<Sp1Fp> = x.iter().map(|e| ark_to_sp1(*e)).collect();

        self.sponge.absorb(&inputs);
    }

    fn absorb_g(&mut self, g: &[Affine<P>]) {
        self.last_squeezed.clear();

        let zero = P::BaseField::zero();
        let mut inputs = Vec::with_capacity(2 * g.len());

        for point in g.iter() {
            if point.infinity {
                inputs.push(ark_to_sp1(zero));
                inputs.push(ark_to_sp1(zero));
            } else {
                inputs.push(ark_to_sp1(point.x));
                inputs.push(ark_to_sp1(point.y));
            }
        }

        self.sponge.absorb(&inputs);
    }

    fn absorb_fr(&mut self, x: &[P::ScalarField]) {
        self.last_squeezed.clear();

        if <P::ScalarField as PrimeField>::MODULUS < <P::BaseField as PrimeField>::MODULUS.into() {
            let mut inputs = Vec::with_capacity(x.len());

            for scalar in x.iter() {
                let bits = scalar.into_bigint().to_bits_le();

                let fq = P::BaseField::from_bigint(
                    <P::BaseField as PrimeField>::BigInt::from_bits_le(&bits),
                )
                .expect("scalar to base field conversion failed");

                inputs.push(ark_to_sp1(fq));
            }

            self.sponge.absorb(&inputs);
        } else {
            let mut inputs = Vec::with_capacity(2 * x.len());

            for scalar in x.iter() {
                let bits = scalar.into_bigint().to_bits_le();

                let low_bit = if bits[0] {
                    P::BaseField::one()
                } else {
                    P::BaseField::zero()
                };

                let high_bits = P::BaseField::from_bigint(
                    <P::BaseField as PrimeField>::BigInt::from_bits_le(&bits[1..]),
                )
                .expect("scalar high bits conversion failed");

                inputs.push(ark_to_sp1(high_bits));
                inputs.push(ark_to_sp1(low_bit));
            }

            self.sponge.absorb(&inputs);
        }
    }

    fn challenge_fq(&mut self) -> P::BaseField {
        self.squeeze_field()
    }

    fn challenge(&mut self) -> P::ScalarField {
        self.squeeze(CHALLENGE_LENGTH_IN_LIMBS)
    }

    fn digest_fq(mut self) -> P::BaseField {
        self.squeeze_field()
    }

    fn digest(mut self) -> P::ScalarField {
        let x: <P::BaseField as PrimeField>::BigInt = self.squeeze_field().into_bigint();

        P::ScalarField::from_bigint(x.into()).unwrap_or_else(P::ScalarField::zero)
    }
}

// -----------------------------------------------------------------------------
// Fr sponge wrapper
// -----------------------------------------------------------------------------

pub struct Sp1FrSponge<Fr: PrimeField, SC = (), const FULL_ROUNDS: usize = 55> {
    pub sponge: Sp1Sponge,
    pub last_squeezed: Vec<u64>,
    _phantom: PhantomData<(Fr, SC)>,
}

impl<Fr, SC, const FULL_ROUNDS: usize>
    From<&'static ArithmeticSpongeParams<Fr, FULL_ROUNDS>>
    for Sp1FrSponge<Fr, SC, FULL_ROUNDS>
where
    Fr: PrimeField + CanonicalSerialize + CanonicalDeserialize,
{
    fn from(_params: &'static ArithmeticSpongeParams<Fr, FULL_ROUNDS>) -> Self {
        Self {
            sponge: Sp1Sponge::new(),
            last_squeezed: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

impl<Fr, SC, const FULL_ROUNDS: usize> Sp1FrSponge<Fr, SC, FULL_ROUNDS>
where
    Fr: PrimeField + CanonicalSerialize + CanonicalDeserialize,
{
    #[inline(always)]
    fn refill_limbs(&mut self) {
        let x: Fr = sp1_to_ark(self.sponge.squeeze());
        let bigint = x.into_bigint();

        self.last_squeezed
            .extend_from_slice(&bigint.as_ref()[0..CHALLENGE_LENGTH_IN_LIMBS]);
    }

    #[inline(always)]
    pub fn squeeze(&mut self, num_limbs: usize) -> Fr {
        while self.last_squeezed.len() < num_limbs {
            self.refill_limbs();
        }

        Fr::from_bigint(pack::<Fr::BigInt>(&take_first_limbs(
            &mut self.last_squeezed,
            num_limbs,
        )))
        .expect("squeezed value is not a valid scalar field element")
    }
}