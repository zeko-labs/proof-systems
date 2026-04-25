// mina-poseidon/src/sp1/mod.rs

mod fp;
mod params;
mod poseidon;

use alloc::vec::Vec;
use crypto_bigint::U256;
use fp::Fp as Sp1Fp;
use poseidon::Sponge as Sp1Sponge;

use crate::{
    constants::SpongeConstants,
    poseidon::ArithmeticSpongeParams,
    sponge::{FqSponge, ScalarChallenge, CHALLENGE_LENGTH_IN_LIMBS},
};
use ark_ec::models::short_weierstrass::{Affine, SWCurveConfig};
use ark_ff::{BigInteger, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

// ---------------------------------------------------------------------------
// Conversions ark ↔ Sp1Fp
// ---------------------------------------------------------------------------

#[inline(always)]
fn ark_to_sp1<F: ark_ff::PrimeField>(x: F) -> Sp1Fp {
    let limbs: [u64; 4] = unsafe { *(x.into_bigint().as_ref().as_ptr() as *const [u64; 4]) };
    Sp1Fp::from_le_limbs(limbs)
}

#[inline(always)]
fn sp1_to_ark<F: ark_ff::PrimeField>(x: Sp1Fp) -> F {
    let limbs = x.to_le_limbs();
    let mut bi = F::BigInt::default();
    bi.as_mut().copy_from_slice(&limbs);
    F::from_bigint(bi).unwrap()
}

// ---------------------------------------------------------------------------
// FqSponge wrapper
// ---------------------------------------------------------------------------

use core::marker::PhantomData;

#[derive(Clone)]
pub struct Sp1FqSponge<P: SWCurveConfig, SC = (), const FULL_ROUNDS: usize = 55> {
    inner: Sp1Sponge,
    last_squeezed: alloc::vec::Vec<u64>,
    _phantom: PhantomData<(P, SC)>,
}

pub struct Sp1FrSponge<Fr: PrimeField, SC = (), const FULL_ROUNDS: usize = 55> {
    inner: Sp1Sponge,
    last_squeezed: alloc::vec::Vec<u64>,
    _phantom: PhantomData<(Fr, SC)>,
}

impl<P: SWCurveConfig> Sp1FqSponge<P>
where
    P::BaseField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    P::ScalarField: PrimeField,
    <P::BaseField as PrimeField>::BigInt: Into<<P::ScalarField as PrimeField>::BigInt>,
{
    fn refill_limbs(&mut self) {
        let x: P::BaseField = sp1_to_ark(self.inner.squeeze());
        let bigint = x.into_bigint();
        self.last_squeezed.extend_from_slice(&bigint.as_ref()[0..2]);
    }

    fn squeeze_limbs(&mut self, num_limbs: usize) -> Vec<u64> {
        while self.last_squeezed.len() < num_limbs {
            self.refill_limbs();
        }
        let out = self.last_squeezed[..num_limbs].to_vec();
        self.last_squeezed = self.last_squeezed[num_limbs..].to_vec();
        out
    }

    fn squeeze_scalar(&mut self, num_limbs: usize) -> P::ScalarField {
        let limbs = self.squeeze_limbs(num_limbs);
        let mut res = <P::ScalarField as PrimeField>::BigInt::from(0u64);
        for &x in limbs.iter().rev() {
            res <<= 64;
            res.add_with_carry(&x.into());
        }
        P::ScalarField::from_bigint(res).expect("squeeze_scalar failed")
    }
}

impl<P: SWCurveConfig, const FULL_ROUNDS: usize>
    FqSponge<P::BaseField, Affine<P>, P::ScalarField, FULL_ROUNDS> for Sp1FqSponge<P>
where
    P::BaseField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    P::ScalarField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    <P::BaseField as PrimeField>::BigInt: Into<<P::ScalarField as PrimeField>::BigInt>,
{
    fn new(_params: &'static ArithmeticSpongeParams<P::BaseField, FULL_ROUNDS>) -> Self {
        Self {
            inner: Sp1Sponge::new(),
            last_squeezed: alloc::vec::Vec::new(),
            _phantom: core::marker::PhantomData,
        }
    }

    fn absorb_fq(&mut self, x: &[P::BaseField]) {
        self.last_squeezed.clear();
        let inputs: alloc::vec::Vec<Sp1Fp> = x.iter().map(|e| ark_to_sp1(*e)).collect();
        self.inner.absorb(&inputs);
    }

    fn absorb_g(&mut self, g: &[Affine<P>]) {
        self.last_squeezed.clear();
        let zero = P::BaseField::zero();
        let mut inputs = alloc::vec::Vec::with_capacity(2 * g.len());
        for point in g.iter() {
            if point.infinity {
                inputs.push(ark_to_sp1(zero));
                inputs.push(ark_to_sp1(zero));
            } else {
                inputs.push(ark_to_sp1(point.x));
                inputs.push(ark_to_sp1(point.y));
            }
        }
        self.inner.absorb(&inputs);
    }

    fn absorb_fr(&mut self, x: &[P::ScalarField]) {
        self.last_squeezed.clear();

        if <P::ScalarField as PrimeField>::MODULUS < <P::BaseField as PrimeField>::MODULUS.into() {
            let mut inputs = alloc::vec::Vec::with_capacity(x.len());
            for scalar in x.iter() {
                let bits = scalar.into_bigint().to_bits_le();
                let fe = P::BaseField::from_bigint(
                    <P::BaseField as PrimeField>::BigInt::from_bits_le(&bits),
                )
                .expect("absorb_fr conversion failed");
                inputs.push(ark_to_sp1(fe));
            }
            self.inner.absorb(&inputs);
        } else {
            let mut inputs = alloc::vec::Vec::with_capacity(2 * x.len());
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
                .expect("absorb_fr high_bits failed");
                inputs.push(ark_to_sp1(high_bits));
                inputs.push(ark_to_sp1(low_bit));
            }
            self.inner.absorb(&inputs);
        }
    }

    fn challenge_fq(&mut self) -> P::BaseField {
        self.last_squeezed.clear();
        sp1_to_ark(self.inner.squeeze())
    }

    fn challenge(&mut self) -> P::ScalarField {
        self.squeeze_scalar(CHALLENGE_LENGTH_IN_LIMBS)
    }

    fn digest_fq(mut self) -> P::BaseField {
        self.last_squeezed.clear();
        sp1_to_ark(self.inner.squeeze())
    }

    fn digest(mut self) -> P::ScalarField {
        let x: <P::BaseField as PrimeField>::BigInt =
            sp1_to_ark::<P::BaseField>(self.inner.squeeze()).into_bigint();
        P::ScalarField::from_bigint(x.into()).unwrap_or_else(P::ScalarField::zero)
    }
}

// ---------------------------------------------------------------------------
// FrSponge wrapper
// ---------------------------------------------------------------------------

// pub struct Sp1FrSponge<Fr: PrimeField> {
//     inner: Sp1Sponge,
//     last_squeezed: Vec<u64>,
//     _phantom: core::marker::PhantomData<Fr>,
// }

// impl<Fr: PrimeField + CanonicalSerialize + CanonicalDeserialize>
//     From<&'static ArithmeticSpongeParams<Fr, 55>> for Sp1FrSponge<Fr>
// {
//     fn from(_params: &'static ArithmeticSpongeParams<Fr, 55>) -> Self {
//         Self {
//             inner: Sp1Sponge::new(),
//             last_squeezed: Vec::new(),
//             _phantom: core::marker::PhantomData,
//         }
//     }
// }

// impl<Fr: PrimeField + CanonicalSerialize + CanonicalDeserialize> crate::sponge::FrSponge<Fr>
//     for Sp1FrSponge<Fr>
// {
//     fn new(_params: &'static ArithmeticSpongeParams<Fr, 55>) -> Self {
//         Self {
//             inner: Sp1Sponge::new(),
//             last_squeezed: Vec::new(),
//             _phantom: core::marker::PhantomData,
//         }
//     }

//     fn absorb(&mut self, x: &Fr) {
//         self.inner.absorb(&[ark_to_sp1(*x)]);
//     }

//     fn challenge(&mut self) -> ScalarChallenge<Fr> {
//         ScalarChallenge(self.squeeze(CHALLENGE_LENGTH_IN_LIMBS))
//     }

//     fn absorb_evaluations(&mut self, e: &crate::sponge::PointEvaluations<alloc::vec::Vec<Fr>>) {
//         for x in e.zeta.iter().chain(e.zeta_omega.iter()) {
//             self.absorb(x);
//         }
//     }
// }

// impl<Fr: PrimeField + CanonicalSerialize + CanonicalDeserialize> Sp1FrSponge<Fr> {
//     fn refill_limbs(&mut self) {
//         let x: Fr = sp1_to_ark(self.inner.squeeze());
//         let bigint = x.into_bigint();
//         self.last_squeezed.extend_from_slice(&bigint.as_ref()[0..2]);
//     }

//     fn squeeze(&mut self, num_limbs: usize) -> Fr {
//         while self.last_squeezed.len() < num_limbs {
//             self.refill_limbs();
//         }
//         let limbs = self.last_squeezed[..num_limbs].to_vec();
//         self.last_squeezed = self.last_squeezed[num_limbs..].to_vec();
//         let mut res = <Fr as PrimeField>::BigInt::from(0u64);
//         for &x in limbs.iter().rev() {
//             res <<= 64;
//             res.add_with_carry(&x.into());
//         }
//         Fr::from_bigint(res).expect("Fr squeeze failed")
//     }
// }
