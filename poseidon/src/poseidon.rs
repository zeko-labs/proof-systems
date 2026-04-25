//! This module implements Poseidon Hash Function primitive

extern crate alloc;

use crate::{
    constants::SpongeConstants,
    permutation::{full_round, poseidon_block_cipher},
};
use alloc::{vec, vec::Vec};
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[cfg(target_os = "zkvm")]
const KIMCHI_FULL_ROUNDS: usize = 55;

#[cfg(target_os = "zkvm")]
const PALLAS_BASE_MODULUS: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
];

#[cfg(target_os = "zkvm")]
const VESTA_BASE_MODULUS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
];

/// Cryptographic sponge interface - for hashing an arbitrary amount of
/// data into one or more field elements
pub trait Sponge<Input: Field, Digest, const FULL_ROUNDS: usize> {
    /// Create a new cryptographic sponge using arithmetic sponge `params`
    fn new(params: &'static ArithmeticSpongeParams<Input, FULL_ROUNDS>) -> Self;

    /// Absorb an array of field elements `x`
    fn absorb(&mut self, x: &[Input]);

    /// Squeeze an output from the sponge
    fn squeeze(&mut self) -> Digest;

    /// Reset the sponge back to its initial state (as if it were just created)
    fn reset(&mut self);
}

pub fn sbox<F: Field, SC: SpongeConstants>(mut x: F) -> F {
    if SC::PERM_SBOX == 7 {
        // This is much faster than using the generic `pow`. Hard-code to get the ~50% speed-up
        // that it gives to hashing.
        let mut square = x;
        square.square_in_place();
        x *= square;
        square.square_in_place();
        x *= square;
        x
    } else {
        x.pow([SC::PERM_SBOX as u64])
    }
}

#[derive(Clone, Debug)]
pub enum SpongeState {
    Absorbed(usize),
    Squeezed(usize),
}

#[derive(Clone, Debug)]
pub struct ArithmeticSpongeParams<
    F: Field + CanonicalSerialize + CanonicalDeserialize,
    const FULL_ROUNDS: usize,
> {
    pub round_constants: [[F; 3]; FULL_ROUNDS],
    pub mds: [[F; 3]; 3],
}

#[cfg(target_os = "zkvm")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PastaFieldKind {
    PallasFp,
    VestaFq,
}

#[cfg(target_os = "zkvm")]
#[derive(Clone, Debug)]
struct Sp1StateCache {
    field_kind: PastaFieldKind,
    modulus: [u64; 4],
    state: [[u64; 4]; 3],
}

#[derive(Clone)]
pub struct ArithmeticSponge<F: Field, SC: SpongeConstants, const FULL_ROUNDS: usize> {
    pub sponge_state: SpongeState,
    rate: usize,
    // TODO(mimoo: an array enforcing the width is better no? or at least an assert somewhere)
    pub state: Vec<F>,
    params: &'static ArithmeticSpongeParams<F, FULL_ROUNDS>,
    pub constants: core::marker::PhantomData<SC>,
    #[cfg(target_os = "zkvm")]
    sp1_cache: Option<Sp1StateCache>,
}

#[cfg(target_os = "zkvm")]
#[inline(always)]
fn detect_pasta_field<F: PrimeField>() -> Option<(PastaFieldKind, [u64; 4])> {
    let ch = F::characteristic();
    let modulus = [
        ch.get(0).copied().unwrap_or(0),
        ch.get(1).copied().unwrap_or(0),
        ch.get(2).copied().unwrap_or(0),
        ch.get(3).copied().unwrap_or(0),
    ];

    match modulus {
        PALLAS_BASE_MODULUS => Some((PastaFieldKind::PallasFp, PALLAS_BASE_MODULUS)),
        VESTA_BASE_MODULUS => Some((PastaFieldKind::VestaFq, VESTA_BASE_MODULUS)),
        _ => None,
    }
}

impl<F: PrimeField, SC: SpongeConstants, const FULL_ROUNDS: usize>
    ArithmeticSponge<F, SC, FULL_ROUNDS>
{
    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn maybe_new_sp1_cache() -> Option<Sp1StateCache> {
        if FULL_ROUNDS != KIMCHI_FULL_ROUNDS {
            return None;
        }

        detect_pasta_field::<F>().map(|(field_kind, modulus)| Sp1StateCache {
            field_kind,
            modulus,
            state: [[0u64; 4]; 3],
        })
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn sync_cache_from_state(&mut self) {
        if let Some(cache) = self.sp1_cache.as_mut() {
            debug_assert!(self.state.len() >= 3);
            for i in 0..3 {
                cache.state[i] = zkvm_fast::from_ark(self.state[i]).0;
            }
        }
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn sync_state_from_cache(&mut self) {
        if let Some(cache) = self.sp1_cache.as_ref() {
            debug_assert!(self.state.len() >= 3);
            for i in 0..3 {
                self.state[i] = zkvm_fast::to_ark(zkvm_fast::Sp1Fp(cache.state[i]));
            }
        }
    }

    #[inline(always)]
    fn add_to_state_slot(&mut self, idx: usize, x: F) {
        self.state[idx].add_assign(&x);

        #[cfg(target_os = "zkvm")]
        if let Some(cache) = self.sp1_cache.as_mut() {
            let x_limbs = zkvm_fast::from_ark(x);
            let cur = zkvm_fast::Sp1Fp(cache.state[idx]);
            cache.state[idx] = zkvm_fast::add(cur, x_limbs, cache.modulus).0;
        }
    }

    pub fn full_round(&mut self, r: usize) {
        full_round::<F, SC, FULL_ROUNDS>(self.params, &mut self.state, r);

        #[cfg(target_os = "zkvm")]
        self.sync_cache_from_state();
    }

    pub fn poseidon_block_cipher(&mut self) {
        #[cfg(target_os = "zkvm")]
        if let Some(cache) = self.sp1_cache.as_mut() {
            zkvm_fast::permute_state::<F, SC, FULL_ROUNDS>(
                &mut cache.state,
                cache.field_kind,
                cache.modulus,
            );
            self.sync_state_from_cache();
            return;
        }

        poseidon_block_cipher::<F, SC, FULL_ROUNDS>(self.params, &mut self.state);

        #[cfg(target_os = "zkvm")]
        self.sync_cache_from_state();
    }
}

impl<F: PrimeField, SC: SpongeConstants, const FULL_ROUNDS: usize> Sponge<F, F, FULL_ROUNDS>
    for ArithmeticSponge<F, SC, FULL_ROUNDS>
{
    fn new(params: &'static ArithmeticSpongeParams<F, FULL_ROUNDS>) -> Self {
        let capacity = SC::SPONGE_CAPACITY;
        let rate = SC::SPONGE_RATE;

        let mut state = Vec::with_capacity(capacity + rate);

        for _ in 0..(capacity + rate) {
            state.push(F::zero());
        }

        Self {
            state,
            rate,
            sponge_state: SpongeState::Absorbed(0),
            params,
            constants: core::marker::PhantomData,
            #[cfg(target_os = "zkvm")]
            sp1_cache: Self::maybe_new_sp1_cache(),
        }
    }

    fn absorb(&mut self, x: &[F]) {
        for x in x.iter().copied() {
            match self.sponge_state {
                SpongeState::Absorbed(n) => {
                    if n == self.rate {
                        self.poseidon_block_cipher();
                        self.sponge_state = SpongeState::Absorbed(1);
                        self.add_to_state_slot(0, x);
                    } else {
                        self.sponge_state = SpongeState::Absorbed(n + 1);
                        self.add_to_state_slot(n, x);
                    }
                }
                SpongeState::Squeezed(_) => {
                    self.add_to_state_slot(0, x);
                    self.sponge_state = SpongeState::Absorbed(1);
                }
            }
        }
    }

    fn squeeze(&mut self) -> F {
        match self.sponge_state {
            SpongeState::Squeezed(n) => {
                if n == self.rate {
                    self.poseidon_block_cipher();
                    self.sponge_state = SpongeState::Squeezed(1);
                    self.state[0]
                } else {
                    self.sponge_state = SpongeState::Squeezed(n + 1);
                    self.state[n]
                }
            }
            SpongeState::Absorbed(_) => {
                self.poseidon_block_cipher();
                self.sponge_state = SpongeState::Squeezed(1);
                self.state[0]
            }
        }
    }

    fn reset(&mut self) {
        self.state = vec![F::zero(); self.state.len()];
        self.sponge_state = SpongeState::Absorbed(0);

        #[cfg(target_os = "zkvm")]
        if let Some(cache) = self.sp1_cache.as_mut() {
            cache.state = [[0u64; 4]; 3];
        }
    }
}

#[cfg(target_os = "zkvm")]
mod zkvm_fast {
    use super::*;
    use crate::pasta::{fp_sp1, fq_sp1};
    use ark_ff::PrimeField;

    type Sp1Limbs = [u64; 4];

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub(crate) struct Sp1Fp(pub(crate) Sp1Limbs);

    #[inline(always)]
    pub(crate) fn from_ark<F: PrimeField>(x: F) -> Sp1Fp {
        let bigint = x.into_bigint();
        let limbs = bigint.as_ref();
        debug_assert!(limbs.len() == 4);
        Sp1Fp([limbs[0], limbs[1], limbs[2], limbs[3]])
    }

    #[inline(always)]
    pub(crate) fn to_ark<F: PrimeField>(x: Sp1Fp) -> F {
        let bytes: [u8; 32] = bytemuck::cast(x.0);
        F::from_le_bytes_mod_order(&bytes)
    }

    #[inline(always)]
    pub(crate) fn add(a: Sp1Fp, b: Sp1Fp, modulus: Sp1Limbs) -> Sp1Fp {
        let mut carry = 0u64;
        let mut out = [0u64; 4];

        for i in 0..4 {
            let (s1, c1) = a.0[i].overflowing_add(b.0[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            out[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }

        let need_reduce = carry != 0 || {
            let mut ge = true;
            for i in (0..4).rev() {
                if out[i] > modulus[i] {
                    break;
                }
                if out[i] < modulus[i] {
                    ge = false;
                    break;
                }
            }
            ge
        };

        if need_reduce {
            let mut borrow = 0u64;
            for i in 0..4 {
                let (d1, b1) = out[i].overflowing_sub(modulus[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                out[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
            }
        }

        Sp1Fp(out)
    }

    #[inline(always)]
    fn mul(a: Sp1Fp, b: Sp1Fp, modulus: Sp1Limbs) -> Sp1Fp {
        let mut out = [0u64; 4];
        #[allow(unsafe_code)]
        unsafe {
            sp1_lib::sys_bigint(&mut out, 0, &a.0, &b.0, &modulus);
        }
        Sp1Fp(out)
    }

    #[inline(always)]
    fn pow7(x: Sp1Fp, modulus: Sp1Limbs) -> Sp1Fp {
        let x2 = mul(x, x, modulus);
        let x4 = mul(x2, x2, modulus);
        let x6 = mul(x4, x2, modulus);
        mul(x6, x, modulus)
    }

    #[inline(always)]
    fn apply_mds_matrix_sp1<SC: SpongeConstants>(
        mds: &[[Sp1Limbs; 3]; 3],
        state: &mut [Sp1Fp; 3],
        modulus: Sp1Limbs,
    ) {
        // Fast path for the special MDS shape.
        if !SC::PERM_FULL_MDS {
            let s0 = state[0];
            let s1 = state[1];
            let s2 = state[2];

            state[0] = add(s0, s2, modulus);
            state[1] = add(s0, s1, modulus);
            state[2] = add(s1, s2, modulus);
            return;
        }

        let tmp = *state;

        state[0] = add(
            add(
                mul(Sp1Fp(mds[0][0]), tmp[0], modulus),
                mul(Sp1Fp(mds[0][1]), tmp[1], modulus),
                modulus,
            ),
            mul(Sp1Fp(mds[0][2]), tmp[2], modulus),
            modulus,
        );

        state[1] = add(
            add(
                mul(Sp1Fp(mds[1][0]), tmp[0], modulus),
                mul(Sp1Fp(mds[1][1]), tmp[1], modulus),
                modulus,
            ),
            mul(Sp1Fp(mds[1][2]), tmp[2], modulus),
            modulus,
        );

        state[2] = add(
            add(
                mul(Sp1Fp(mds[2][0]), tmp[0], modulus),
                mul(Sp1Fp(mds[2][1]), tmp[1], modulus),
                modulus,
            ),
            mul(Sp1Fp(mds[2][2]), tmp[2], modulus),
            modulus,
        );
    }

    #[inline(always)]
    fn full_round_sp1<SC: SpongeConstants>(
        state: &mut [Sp1Fp; 3],
        mds: &[[Sp1Limbs; 3]; 3],
        rc: &[Sp1Limbs; 3],
        modulus: Sp1Limbs,
    ) {
        state[0] = pow7(state[0], modulus);
        state[1] = pow7(state[1], modulus);
        state[2] = pow7(state[2], modulus);

        apply_mds_matrix_sp1::<SC>(mds, state, modulus);

        state[0] = add(state[0], Sp1Fp(rc[0]), modulus);
        state[1] = add(state[1], Sp1Fp(rc[1]), modulus);
        state[2] = add(state[2], Sp1Fp(rc[2]), modulus);
    }

    #[inline(always)]
    fn half_rounds_sp1<SC: SpongeConstants>(
        mds: &[[Sp1Limbs; 3]; 3],
        rc: &[[Sp1Limbs; 3]; KIMCHI_FULL_ROUNDS],
        state: &mut [Sp1Fp; 3],
        modulus: Sp1Limbs,
    ) {
        for r in 0..SC::PERM_HALF_ROUNDS_FULL {
            state[0] = add(state[0], Sp1Fp(rc[r][0]), modulus);
            state[1] = add(state[1], Sp1Fp(rc[r][1]), modulus);
            state[2] = add(state[2], Sp1Fp(rc[r][2]), modulus);

            state[0] = pow7(state[0], modulus);
            state[1] = pow7(state[1], modulus);
            state[2] = pow7(state[2], modulus);

            apply_mds_matrix_sp1::<SC>(mds, state, modulus);
        }

        for r in 0..SC::PERM_ROUNDS_PARTIAL {
            let rr = SC::PERM_HALF_ROUNDS_FULL + r;

            state[0] = add(state[0], Sp1Fp(rc[rr][0]), modulus);
            state[1] = add(state[1], Sp1Fp(rc[rr][1]), modulus);
            state[2] = add(state[2], Sp1Fp(rc[rr][2]), modulus);

            state[0] = pow7(state[0], modulus);

            apply_mds_matrix_sp1::<SC>(mds, state, modulus);
        }

        for r in 0..SC::PERM_HALF_ROUNDS_FULL {
            let rr = SC::PERM_HALF_ROUNDS_FULL + SC::PERM_ROUNDS_PARTIAL + r;

            state[0] = add(state[0], Sp1Fp(rc[rr][0]), modulus);
            state[1] = add(state[1], Sp1Fp(rc[rr][1]), modulus);
            state[2] = add(state[2], Sp1Fp(rc[rr][2]), modulus);

            state[0] = pow7(state[0], modulus);
            state[1] = pow7(state[1], modulus);
            state[2] = pow7(state[2], modulus);

            apply_mds_matrix_sp1::<SC>(mds, state, modulus);
        }
    }

    #[inline(always)]
    fn permute_with_constants<F: PrimeField + Copy, SC: SpongeConstants>(
        state: &mut [[u64; 4]; 3],
        mds: &[[Sp1Limbs; 3]; 3],
        rc: &[[Sp1Limbs; 3]; KIMCHI_FULL_ROUNDS],
        modulus: Sp1Limbs,
    ) {
        let mut s = [Sp1Fp(state[0]), Sp1Fp(state[1]), Sp1Fp(state[2])];

        if SC::PERM_HALF_ROUNDS_FULL == 0 {
            if SC::PERM_INITIAL_ARK {
                s[0] = add(s[0], Sp1Fp(rc[0][0]), modulus);
                s[1] = add(s[1], Sp1Fp(rc[0][1]), modulus);
                s[2] = add(s[2], Sp1Fp(rc[0][2]), modulus);

                for r in 0..SC::PERM_ROUNDS_FULL {
                    full_round_sp1::<SC>(&mut s, mds, &rc[r + 1], modulus);
                }
            } else {
                for r in 0..SC::PERM_ROUNDS_FULL {
                    full_round_sp1::<SC>(&mut s, mds, &rc[r], modulus);
                }
            }
        } else {
            half_rounds_sp1::<SC>(mds, rc, &mut s, modulus);
        }

        state[0] = s[0].0;
        state[1] = s[1].0;
        state[2] = s[2].0;
    }

    pub(crate) fn permute_state<F: PrimeField + Copy, SC: SpongeConstants, const FULL_ROUNDS: usize>(
        state: &mut [[u64; 4]; 3],
        field_kind: PastaFieldKind,
        modulus: Sp1Limbs,
    ) {
        if FULL_ROUNDS != KIMCHI_FULL_ROUNDS {
            return;
        }

        match field_kind {
            PastaFieldKind::PallasFp => {
                permute_with_constants::<F, SC>(state, &fp_sp1::MDS, &fp_sp1::ROUND_CONSTANTS, modulus);
            }
            PastaFieldKind::VestaFq => {
                permute_with_constants::<F, SC>(state, &fq_sp1::MDS, &fq_sp1::ROUND_CONSTANTS, modulus);
            }
        }
    }
}