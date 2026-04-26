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
macro_rules! zk_cycle_start {
    ($name:expr) => {
        // std::println!(concat!("cycle-tracker-start: ", $name));
    };
}

#[cfg(target_os = "zkvm")]
macro_rules! zk_cycle_end {
    ($name:expr) => {
        // std::println!(concat!("cycle-tracker-end: ", $name));
    };
}

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

/// Cryptographic sponge interface for hashing an arbitrary amount of
/// data into one or more field elements.
pub trait Sponge<Input: Field, Digest, const FULL_ROUNDS: usize> {
    /// Create a new cryptographic sponge using arithmetic sponge params.
    fn new(params: &'static ArithmeticSpongeParams<Input, FULL_ROUNDS>) -> Self;

    /// Absorb an array of field elements.
    fn absorb(&mut self, x: &[Input]);

    /// Squeeze an output from the sponge.
    fn squeeze(&mut self) -> Digest;

    /// Reset the sponge back to its initial state.
    fn reset(&mut self);
}

pub fn sbox<F: Field, SC: SpongeConstants>(mut x: F) -> F {
    if SC::PERM_SBOX == 7 {
        // This is much faster than using the generic `pow`.
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

#[cfg(target_os = "zkvm")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastKimchiPhase {
    Absorbed0,
    Absorbed1,
    Absorbed2,
    Squeezed1,
    Squeezed2,
}

#[derive(Clone)]
pub struct ArithmeticSponge<
    F: Field + CanonicalSerialize + CanonicalDeserialize,
    SC: SpongeConstants,
    const FULL_ROUNDS: usize,
> {
    pub sponge_state: SpongeState,
    rate: usize,
    pub state: Vec<F>,
    params: &'static ArithmeticSpongeParams<F, FULL_ROUNDS>,
    pub constants: core::marker::PhantomData<SC>,
    #[cfg(target_os = "zkvm")]
    sp1_cache: Option<Sp1StateCache>,
    #[cfg(target_os = "zkvm")]
    sp1_state_stale: bool,
    #[cfg(target_os = "zkvm")]
    fast_kimchi_phase: Option<FastKimchiPhase>,
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

impl<
        F: PrimeField + CanonicalSerialize + CanonicalDeserialize,
        SC: SpongeConstants,
        const FULL_ROUNDS: usize,
    > ArithmeticSponge<F, SC, FULL_ROUNDS>
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
    fn maybe_new_fast_phase(rate: usize, has_cache: bool) -> Option<FastKimchiPhase> {
        if has_cache && FULL_ROUNDS == KIMCHI_FULL_ROUNDS && rate == 2 {
            Some(FastKimchiPhase::Absorbed0)
        } else {
            None
        }
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn has_fast_kimchi_path(&self) -> bool {
        self.fast_kimchi_phase.is_some()
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn set_fast_phase(&mut self, phase: FastKimchiPhase) {
        self.fast_kimchi_phase = Some(phase);
        self.sponge_state = match phase {
            FastKimchiPhase::Absorbed0 => SpongeState::Absorbed(0),
            FastKimchiPhase::Absorbed1 => SpongeState::Absorbed(1),
            FastKimchiPhase::Absorbed2 => SpongeState::Absorbed(2),
            FastKimchiPhase::Squeezed1 => SpongeState::Squeezed(1),
            FastKimchiPhase::Squeezed2 => SpongeState::Squeezed(2),
        };
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn sync_cache_from_state(&mut self) {
        if let Some(cache) = self.sp1_cache.as_mut() {
            debug_assert!(self.state.len() >= 3);
            cache.state[0] = zkvm_fast::from_ark(self.state[0]).0;
            cache.state[1] = zkvm_fast::from_ark(self.state[1]).0;
            cache.state[2] = zkvm_fast::from_ark(self.state[2]).0;
            self.sp1_state_stale = false;
        }
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn ensure_state_synced_from_cache(&mut self) {
        if !self.sp1_state_stale {
            return;
        }

        if let Some(cache) = self.sp1_cache.as_ref() {
            debug_assert!(self.state.len() >= 3);
            self.state[0] = zkvm_fast::to_ark::<F>(zkvm_fast::Sp1Fp(cache.state[0]));
            self.state[1] = zkvm_fast::to_ark::<F>(zkvm_fast::Sp1Fp(cache.state[1]));
            self.state[2] = zkvm_fast::to_ark::<F>(zkvm_fast::Sp1Fp(cache.state[2]));
            self.sp1_state_stale = false;
        }
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn read_cache_slot(&self, idx: usize) -> F {
        let cache = self.sp1_cache.as_ref().unwrap();
        zkvm_fast::to_ark::<F>(zkvm_fast::Sp1Fp(cache.state[idx]))
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn cache_add_to_slot(&mut self, idx: usize, x: F) {
        let cache = self.sp1_cache.as_mut().unwrap();
        let x_limbs = zkvm_fast::from_ark(x);
        let cur = zkvm_fast::Sp1Fp(cache.state[idx]);
        cache.state[idx] = zkvm_fast::add(cur, x_limbs, cache.modulus).0;
        self.sp1_state_stale = true;
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn poseidon_block_cipher_fast(&mut self) {
        let cache = self.sp1_cache.as_mut().unwrap();
        zkvm_fast::permute_state::<SC, FULL_ROUNDS>(
            &mut cache.state,
            cache.field_kind,
            cache.modulus,
        );
        self.sp1_state_stale = true;
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn absorb_fast_kimchi(&mut self, inputs: &[F]) -> bool {
        if !self.has_fast_kimchi_path() {
            return false;
        }

        for x in inputs.iter().copied() {
            match self.fast_kimchi_phase.unwrap() {
                FastKimchiPhase::Absorbed0 => {
                    self.cache_add_to_slot(0, x);
                    self.set_fast_phase(FastKimchiPhase::Absorbed1);
                }
                FastKimchiPhase::Absorbed1 => {
                    self.cache_add_to_slot(1, x);
                    self.set_fast_phase(FastKimchiPhase::Absorbed2);
                }
                FastKimchiPhase::Absorbed2 => {
                    self.poseidon_block_cipher_fast();
                    self.cache_add_to_slot(0, x);
                    self.set_fast_phase(FastKimchiPhase::Absorbed1);
                }
                FastKimchiPhase::Squeezed1 | FastKimchiPhase::Squeezed2 => {
                    self.cache_add_to_slot(0, x);
                    self.set_fast_phase(FastKimchiPhase::Absorbed1);
                }
            }
        }

        true
    }

    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn squeeze_fast_kimchi(&mut self) -> Option<F> {
        if !self.has_fast_kimchi_path() {
            return None;
        }

        match self.fast_kimchi_phase.unwrap() {
            FastKimchiPhase::Absorbed0
            | FastKimchiPhase::Absorbed1
            | FastKimchiPhase::Absorbed2 => {
                self.poseidon_block_cipher_fast();
                self.set_fast_phase(FastKimchiPhase::Squeezed1);
                Some(self.read_cache_slot(0))
            }
            FastKimchiPhase::Squeezed1 => {
                self.set_fast_phase(FastKimchiPhase::Squeezed2);
                Some(self.read_cache_slot(1))
            }
            FastKimchiPhase::Squeezed2 => {
                self.poseidon_block_cipher_fast();
                self.set_fast_phase(FastKimchiPhase::Squeezed1);
                Some(self.read_cache_slot(0))
            }
        }
    }

    #[inline(always)]
    fn add_to_state_slot(&mut self, idx: usize, x: F) {
        #[cfg(target_os = "zkvm")]
        {
            if let Some(cache) = self.sp1_cache.as_mut() {
                let x_limbs = zkvm_fast::from_ark(x);
                let cur = zkvm_fast::Sp1Fp(cache.state[idx]);
                cache.state[idx] = zkvm_fast::add(cur, x_limbs, cache.modulus).0;

                if !self.sp1_state_stale {
                    self.state[idx].add_assign(&x);
                }

                return;
            }
        }

        self.state[idx].add_assign(&x);
    }

    pub fn full_round(&mut self, r: usize) {
        #[cfg(target_os = "zkvm")]
        {
            self.ensure_state_synced_from_cache();
            if self.has_fast_kimchi_path() {
                self.fast_kimchi_phase = None;
            }
        }

        full_round::<F, SC, FULL_ROUNDS>(self.params, &mut self.state, r);

        #[cfg(target_os = "zkvm")]
        self.sync_cache_from_state();
    }

    pub fn poseidon_block_cipher(&mut self) {
        #[cfg(target_os = "zkvm")]
        zk_cycle_start!("poseidon_block_cipher_total");

        #[cfg(target_os = "zkvm")]
        if self.has_fast_kimchi_path() {
            zk_cycle_start!("poseidon_fast_kimchi_permute");
            self.poseidon_block_cipher_fast();
            zk_cycle_end!("poseidon_fast_kimchi_permute");
            zk_cycle_end!("poseidon_block_cipher_total");
            return;
        }

        #[cfg(target_os = "zkvm")]
        if let Some(cache) = self.sp1_cache.as_mut() {
            zk_cycle_start!("poseidon_sp1_cache_permute");

            zkvm_fast::permute_state::<SC, FULL_ROUNDS>(
                &mut cache.state,
                cache.field_kind,
                cache.modulus,
            );

            self.sp1_state_stale = true;

            zk_cycle_end!("poseidon_sp1_cache_permute");
            zk_cycle_end!("poseidon_block_cipher_total");
            return;
        }

        #[cfg(target_os = "zkvm")]
        zk_cycle_start!("poseidon_generic_permute");

        poseidon_block_cipher::<F, SC, FULL_ROUNDS>(self.params, &mut self.state);

        #[cfg(target_os = "zkvm")]
        zk_cycle_end!("poseidon_generic_permute");

        #[cfg(target_os = "zkvm")]
        {
            zk_cycle_start!("poseidon_sync_cache_from_state");
            self.sync_cache_from_state();
            zk_cycle_end!("poseidon_sync_cache_from_state");
            zk_cycle_end!("poseidon_block_cipher_total");
        }
    }
}

impl<
        F: PrimeField + CanonicalSerialize + CanonicalDeserialize,
        SC: SpongeConstants,
        const FULL_ROUNDS: usize,
    > Sponge<F, F, FULL_ROUNDS> for ArithmeticSponge<F, SC, FULL_ROUNDS>
{
    fn new(params: &'static ArithmeticSpongeParams<F, FULL_ROUNDS>) -> Self {
        let capacity = SC::SPONGE_CAPACITY;
        let rate = SC::SPONGE_RATE;

        let mut state = Vec::with_capacity(capacity + rate);
        for _ in 0..(capacity + rate) {
            state.push(F::zero());
        }

        #[cfg(target_os = "zkvm")]
        let sp1_cache = Self::maybe_new_sp1_cache();

        #[cfg(target_os = "zkvm")]
        let fast_kimchi_phase = Self::maybe_new_fast_phase(rate, sp1_cache.is_some());

        Self {
            state,
            rate,
            sponge_state: SpongeState::Absorbed(0),
            params,
            constants: core::marker::PhantomData,
            #[cfg(target_os = "zkvm")]
            sp1_cache,
            #[cfg(target_os = "zkvm")]
            sp1_state_stale: false,
            #[cfg(target_os = "zkvm")]
            fast_kimchi_phase,
        }
    }

    fn absorb(&mut self, x: &[F]) {
        #[cfg(target_os = "zkvm")]
        zk_cycle_start!("sponge_absorb_total");

        #[cfg(target_os = "zkvm")]
        {
            zk_cycle_start!("sponge_absorb_fast_kimchi");
            if self.absorb_fast_kimchi(x) {
                zk_cycle_end!("sponge_absorb_fast_kimchi");
                zk_cycle_end!("sponge_absorb_total");
                return;
            }
            zk_cycle_end!("sponge_absorb_fast_kimchi");
        }

        for x in x.iter().copied() {
            #[cfg(target_os = "zkvm")]
            zk_cycle_start!("sponge_absorb_one");

            match self.sponge_state {
                SpongeState::Absorbed(n) => {
                    if n == self.rate {
                        #[cfg(target_os = "zkvm")]
                        zk_cycle_start!("sponge_absorb_permute");

                        self.poseidon_block_cipher();

                        #[cfg(target_os = "zkvm")]
                        zk_cycle_end!("sponge_absorb_permute");

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

            #[cfg(target_os = "zkvm")]
            zk_cycle_end!("sponge_absorb_one");
        }

        #[cfg(target_os = "zkvm")]
        zk_cycle_end!("sponge_absorb_total");
    }

    fn squeeze(&mut self) -> F {
        #[cfg(target_os = "zkvm")]
        zk_cycle_start!("sponge_squeeze_total");

        #[cfg(target_os = "zkvm")]
        {
            zk_cycle_start!("sponge_squeeze_fast_kimchi");
            if let Some(out) = self.squeeze_fast_kimchi() {
                zk_cycle_end!("sponge_squeeze_fast_kimchi");
                zk_cycle_end!("sponge_squeeze_total");
                return out;
            }
            zk_cycle_end!("sponge_squeeze_fast_kimchi");
        }

        let out = match self.sponge_state {
            SpongeState::Squeezed(n) => {
                if n == self.rate {
                    #[cfg(target_os = "zkvm")]
                    zk_cycle_start!("sponge_squeeze_permute");

                    self.poseidon_block_cipher();

                    #[cfg(target_os = "zkvm")]
                    zk_cycle_end!("sponge_squeeze_permute");

                    self.sponge_state = SpongeState::Squeezed(1);
                    self.state[0]
                } else {
                    self.sponge_state = SpongeState::Squeezed(n + 1);
                    self.state[n]
                }
            }
            SpongeState::Absorbed(_) => {
                #[cfg(target_os = "zkvm")]
                zk_cycle_start!("sponge_squeeze_permute");

                self.poseidon_block_cipher();

                #[cfg(target_os = "zkvm")]
                zk_cycle_end!("sponge_squeeze_permute");

                self.sponge_state = SpongeState::Squeezed(1);
                self.state[0]
            }
        };

        #[cfg(target_os = "zkvm")]
        zk_cycle_end!("sponge_squeeze_total");

        out
    }

    fn reset(&mut self) {
        self.state = vec![F::zero(); self.state.len()];
        self.sponge_state = SpongeState::Absorbed(0);

        #[cfg(target_os = "zkvm")]
        {
            if let Some(cache) = self.sp1_cache.as_mut() {
                cache.state = [[0u64; 4]; 3];
            }
            self.sp1_state_stale = false;
            if self.fast_kimchi_phase.is_some() {
                self.fast_kimchi_phase = Some(FastKimchiPhase::Absorbed0);
            }
        }
    }
}

#[cfg(target_os = "zkvm")]
mod zkvm_fast {
    use super::*;
    use crate::pasta::{fp_sp1, fq_sp1};

    type Sp1Limbs = [u64; 4];

    const PALLAS_M: Sp1Limbs = super::PALLAS_BASE_MODULUS;
    const VESTA_M: Sp1Limbs = super::VESTA_BASE_MODULUS;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub(crate) struct Sp1Fp(pub(crate) Sp1Limbs);

    #[inline(always)]
    pub(crate) fn from_ark<F: PrimeField + CanonicalSerialize>(x: F) -> Sp1Fp {
        let mut buf = [0u8; 32];
        x.serialize_uncompressed(&mut buf[..]).unwrap();
        Sp1Fp(bytemuck::cast(buf))
    }

    #[inline(always)]
    pub(crate) fn to_ark<F: PrimeField + CanonicalDeserialize>(x: Sp1Fp) -> F {
        let buf: [u8; 32] = bytemuck::cast(x.0);
        F::deserialize_uncompressed(&buf[..]).unwrap()
    }

    #[inline(always)]
    pub(crate) fn add(a: Sp1Fp, b: Sp1Fp, m: Sp1Limbs) -> Sp1Fp {
        let mut out = [0u64; 4];
        let mut carry = 0u64;

        for i in 0..4 {
            let (s1, c1) = a.0[i].overflowing_add(b.0[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            out[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }

        let need_reduce = carry != 0 || ge_limbs(out, m);
        if need_reduce {
            out = sub_limbs(out, m);
        }
        Sp1Fp(out)
    }

    #[inline(always)]
    fn ge_limbs(a: Sp1Limbs, b: Sp1Limbs) -> bool {
        for i in (0..4).rev() {
            if a[i] > b[i] {
                return true;
            }
            if a[i] < b[i] {
                return false;
            }
        }
        true
    }

    #[inline(always)]
    fn sub_limbs(mut x: Sp1Limbs, m: Sp1Limbs) -> Sp1Limbs {
        let mut borrow = 0u64;
        for i in 0..4 {
            let (d1, b1) = x[i].overflowing_sub(m[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            x[i] = d2;
            borrow = (b1 as u64) + (b2 as u64);
        }
        x
    }

    #[inline(always)]
    fn add3_reduce(a: Sp1Fp, b: Sp1Fp, c: Sp1Fp, m: Sp1Limbs) -> Sp1Fp {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            let sum = a.0[i] as u128 + b.0[i] as u128 + c.0[i] as u128 + carry as u128;
            out[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        if carry != 0 || ge_limbs(out, m) {
            out = sub_limbs(out, m);
        }
        if ge_limbs(out, m) {
            out = sub_limbs(out, m);
        }
        Sp1Fp(out)
    }

    #[inline(always)]
    fn mul(a: Sp1Fp, b: Sp1Fp, m: Sp1Limbs) -> Sp1Fp {
        let mut out = [0u64; 4];
        #[allow(unsafe_code)]
        unsafe {
            sp1_lib::sys_bigint(&mut out, 0, &a.0, &b.0, &m);
        }
        Sp1Fp(out)
    }

    #[inline(always)]
    fn pow7(x: Sp1Fp, m: Sp1Limbs) -> Sp1Fp {
        let x2 = mul(x, x, m);
        let x4 = mul(x2, x2, m);
        let x6 = mul(x4, x2, m);
        mul(x6, x, m)
    }

    // ---------------------------------------------------------------------------
    // Pallas — toutes les constantes sont statiques
    // ---------------------------------------------------------------------------

    #[inline(always)]
    fn apply_mds_pallas<SC: SpongeConstants>(s: &mut [Sp1Fp; 3]) {
        if !SC::PERM_FULL_MDS {
            let (s0, s1, s2) = (s[0], s[1], s[2]);
            s[0] = add(s0, s2, PALLAS_M);
            s[1] = add(s0, s1, PALLAS_M);
            s[2] = add(s1, s2, PALLAS_M);
            return;
        }
        let (s0, s1, s2) = (s[0], s[1], s[2]);
        s[0] = add3_reduce(
            mul(Sp1Fp(fp_sp1::MDS[0][0]), s0, PALLAS_M),
            mul(Sp1Fp(fp_sp1::MDS[0][1]), s1, PALLAS_M),
            mul(Sp1Fp(fp_sp1::MDS[0][2]), s2, PALLAS_M),
            PALLAS_M,
        );
        s[1] = add3_reduce(
            mul(Sp1Fp(fp_sp1::MDS[1][0]), s0, PALLAS_M),
            mul(Sp1Fp(fp_sp1::MDS[1][1]), s1, PALLAS_M),
            mul(Sp1Fp(fp_sp1::MDS[1][2]), s2, PALLAS_M),
            PALLAS_M,
        );
        s[2] = add3_reduce(
            mul(Sp1Fp(fp_sp1::MDS[2][0]), s0, PALLAS_M),
            mul(Sp1Fp(fp_sp1::MDS[2][1]), s1, PALLAS_M),
            mul(Sp1Fp(fp_sp1::MDS[2][2]), s2, PALLAS_M),
            PALLAS_M,
        );
    }

    #[inline(always)]
    fn full_round_pallas<SC: SpongeConstants>(s: &mut [Sp1Fp; 3], r: usize) {
        s[0] = pow7(s[0], PALLAS_M);
        s[1] = pow7(s[1], PALLAS_M);
        s[2] = pow7(s[2], PALLAS_M);
        apply_mds_pallas::<SC>(s);
        s[0] = add(s[0], Sp1Fp(fp_sp1::ROUND_CONSTANTS[r][0]), PALLAS_M);
        s[1] = add(s[1], Sp1Fp(fp_sp1::ROUND_CONSTANTS[r][1]), PALLAS_M);
        s[2] = add(s[2], Sp1Fp(fp_sp1::ROUND_CONSTANTS[r][2]), PALLAS_M);
    }

    #[inline(always)]
    fn half_rounds_pallas<SC: SpongeConstants>(s: &mut [Sp1Fp; 3]) {
        for r in 0..SC::PERM_HALF_ROUNDS_FULL {
            s[0] = add(s[0], Sp1Fp(fp_sp1::ROUND_CONSTANTS[r][0]), PALLAS_M);
            s[1] = add(s[1], Sp1Fp(fp_sp1::ROUND_CONSTANTS[r][1]), PALLAS_M);
            s[2] = add(s[2], Sp1Fp(fp_sp1::ROUND_CONSTANTS[r][2]), PALLAS_M);
            s[0] = pow7(s[0], PALLAS_M);
            s[1] = pow7(s[1], PALLAS_M);
            s[2] = pow7(s[2], PALLAS_M);
            apply_mds_pallas::<SC>(s);
        }
        for r in 0..SC::PERM_ROUNDS_PARTIAL {
            let rr = SC::PERM_HALF_ROUNDS_FULL + r;
            s[0] = add(s[0], Sp1Fp(fp_sp1::ROUND_CONSTANTS[rr][0]), PALLAS_M);
            s[1] = add(s[1], Sp1Fp(fp_sp1::ROUND_CONSTANTS[rr][1]), PALLAS_M);
            s[2] = add(s[2], Sp1Fp(fp_sp1::ROUND_CONSTANTS[rr][2]), PALLAS_M);
            s[0] = pow7(s[0], PALLAS_M);
            apply_mds_pallas::<SC>(s);
        }
        for r in 0..SC::PERM_HALF_ROUNDS_FULL {
            let rr = SC::PERM_HALF_ROUNDS_FULL + SC::PERM_ROUNDS_PARTIAL + r;
            s[0] = add(s[0], Sp1Fp(fp_sp1::ROUND_CONSTANTS[rr][0]), PALLAS_M);
            s[1] = add(s[1], Sp1Fp(fp_sp1::ROUND_CONSTANTS[rr][1]), PALLAS_M);
            s[2] = add(s[2], Sp1Fp(fp_sp1::ROUND_CONSTANTS[rr][2]), PALLAS_M);
            s[0] = pow7(s[0], PALLAS_M);
            s[1] = pow7(s[1], PALLAS_M);
            s[2] = pow7(s[2], PALLAS_M);
            apply_mds_pallas::<SC>(s);
        }
    }

    #[inline(always)]
    fn permute_pallas<SC: SpongeConstants>(state: &mut [[u64; 4]; 3]) {
        let mut s = [Sp1Fp(state[0]), Sp1Fp(state[1]), Sp1Fp(state[2])];
        if SC::PERM_HALF_ROUNDS_FULL == 0 {
            if SC::PERM_INITIAL_ARK {
                s[0] = add(s[0], Sp1Fp(fp_sp1::ROUND_CONSTANTS[0][0]), PALLAS_M);
                s[1] = add(s[1], Sp1Fp(fp_sp1::ROUND_CONSTANTS[0][1]), PALLAS_M);
                s[2] = add(s[2], Sp1Fp(fp_sp1::ROUND_CONSTANTS[0][2]), PALLAS_M);
                for r in 0..SC::PERM_ROUNDS_FULL {
                    full_round_pallas::<SC>(&mut s, r + 1);
                }
            } else {
                for r in 0..SC::PERM_ROUNDS_FULL {
                    full_round_pallas::<SC>(&mut s, r);
                }
            }
        } else {
            half_rounds_pallas::<SC>(&mut s);
        }
        state[0] = s[0].0;
        state[1] = s[1].0;
        state[2] = s[2].0;
    }

    // ---------------------------------------------------------------------------
    // Vesta — même structure avec fq_sp1 et VESTA_M
    // ---------------------------------------------------------------------------

    #[inline(always)]
    fn apply_mds_vesta<SC: SpongeConstants>(s: &mut [Sp1Fp; 3]) {
        if !SC::PERM_FULL_MDS {
            let (s0, s1, s2) = (s[0], s[1], s[2]);
            s[0] = add(s0, s2, VESTA_M);
            s[1] = add(s0, s1, VESTA_M);
            s[2] = add(s1, s2, VESTA_M);
            return;
        }
        let (s0, s1, s2) = (s[0], s[1], s[2]);
        s[0] = add3_reduce(
            mul(Sp1Fp(fq_sp1::MDS[0][0]), s0, VESTA_M),
            mul(Sp1Fp(fq_sp1::MDS[0][1]), s1, VESTA_M),
            mul(Sp1Fp(fq_sp1::MDS[0][2]), s2, VESTA_M),
            VESTA_M,
        );
        s[1] = add3_reduce(
            mul(Sp1Fp(fq_sp1::MDS[1][0]), s0, VESTA_M),
            mul(Sp1Fp(fq_sp1::MDS[1][1]), s1, VESTA_M),
            mul(Sp1Fp(fq_sp1::MDS[1][2]), s2, VESTA_M),
            VESTA_M,
        );
        s[2] = add3_reduce(
            mul(Sp1Fp(fq_sp1::MDS[2][0]), s0, VESTA_M),
            mul(Sp1Fp(fq_sp1::MDS[2][1]), s1, VESTA_M),
            mul(Sp1Fp(fq_sp1::MDS[2][2]), s2, VESTA_M),
            VESTA_M,
        );
    }

    #[inline(always)]
    fn full_round_vesta<SC: SpongeConstants>(s: &mut [Sp1Fp; 3], r: usize) {
        s[0] = pow7(s[0], VESTA_M);
        s[1] = pow7(s[1], VESTA_M);
        s[2] = pow7(s[2], VESTA_M);
        apply_mds_vesta::<SC>(s);
        s[0] = add(s[0], Sp1Fp(fq_sp1::ROUND_CONSTANTS[r][0]), VESTA_M);
        s[1] = add(s[1], Sp1Fp(fq_sp1::ROUND_CONSTANTS[r][1]), VESTA_M);
        s[2] = add(s[2], Sp1Fp(fq_sp1::ROUND_CONSTANTS[r][2]), VESTA_M);
    }

    #[inline(always)]
    fn half_rounds_vesta<SC: SpongeConstants>(s: &mut [Sp1Fp; 3]) {
        for r in 0..SC::PERM_HALF_ROUNDS_FULL {
            s[0] = add(s[0], Sp1Fp(fq_sp1::ROUND_CONSTANTS[r][0]), VESTA_M);
            s[1] = add(s[1], Sp1Fp(fq_sp1::ROUND_CONSTANTS[r][1]), VESTA_M);
            s[2] = add(s[2], Sp1Fp(fq_sp1::ROUND_CONSTANTS[r][2]), VESTA_M);
            s[0] = pow7(s[0], VESTA_M);
            s[1] = pow7(s[1], VESTA_M);
            s[2] = pow7(s[2], VESTA_M);
            apply_mds_vesta::<SC>(s);
        }
        for r in 0..SC::PERM_ROUNDS_PARTIAL {
            let rr = SC::PERM_HALF_ROUNDS_FULL + r;
            s[0] = add(s[0], Sp1Fp(fq_sp1::ROUND_CONSTANTS[rr][0]), VESTA_M);
            s[1] = add(s[1], Sp1Fp(fq_sp1::ROUND_CONSTANTS[rr][1]), VESTA_M);
            s[2] = add(s[2], Sp1Fp(fq_sp1::ROUND_CONSTANTS[rr][2]), VESTA_M);
            s[0] = pow7(s[0], VESTA_M);
            apply_mds_vesta::<SC>(s);
        }
        for r in 0..SC::PERM_HALF_ROUNDS_FULL {
            let rr = SC::PERM_HALF_ROUNDS_FULL + SC::PERM_ROUNDS_PARTIAL + r;
            s[0] = add(s[0], Sp1Fp(fq_sp1::ROUND_CONSTANTS[rr][0]), VESTA_M);
            s[1] = add(s[1], Sp1Fp(fq_sp1::ROUND_CONSTANTS[rr][1]), VESTA_M);
            s[2] = add(s[2], Sp1Fp(fq_sp1::ROUND_CONSTANTS[rr][2]), VESTA_M);
            s[0] = pow7(s[0], VESTA_M);
            s[1] = pow7(s[1], VESTA_M);
            s[2] = pow7(s[2], VESTA_M);
            apply_mds_vesta::<SC>(s);
        }
    }

    #[inline(always)]
    fn permute_vesta<SC: SpongeConstants>(state: &mut [[u64; 4]; 3]) {
        let mut s = [Sp1Fp(state[0]), Sp1Fp(state[1]), Sp1Fp(state[2])];
        if SC::PERM_HALF_ROUNDS_FULL == 0 {
            if SC::PERM_INITIAL_ARK {
                s[0] = add(s[0], Sp1Fp(fq_sp1::ROUND_CONSTANTS[0][0]), VESTA_M);
                s[1] = add(s[1], Sp1Fp(fq_sp1::ROUND_CONSTANTS[0][1]), VESTA_M);
                s[2] = add(s[2], Sp1Fp(fq_sp1::ROUND_CONSTANTS[0][2]), VESTA_M);
                for r in 0..SC::PERM_ROUNDS_FULL {
                    full_round_vesta::<SC>(&mut s, r + 1);
                }
            } else {
                for r in 0..SC::PERM_ROUNDS_FULL {
                    full_round_vesta::<SC>(&mut s, r);
                }
            }
        } else {
            half_rounds_vesta::<SC>(&mut s);
        }
        state[0] = s[0].0;
        state[1] = s[1].0;
        state[2] = s[2].0;
    }

    // ---------------------------------------------------------------------------
    // Point d'entrée public
    // ---------------------------------------------------------------------------

    pub(crate) fn permute_state<SC: SpongeConstants, const FULL_ROUNDS: usize>(
        state: &mut [[u64; 4]; 3],
        field_kind: PastaFieldKind,
        _modulus: Sp1Limbs, // ignoré — constantes statiques utilisées
    ) {
        if FULL_ROUNDS != KIMCHI_FULL_ROUNDS {
            return;
        }
        match field_kind {
            PastaFieldKind::PallasFp => permute_pallas::<SC>(state),
            PastaFieldKind::VestaFq => permute_vesta::<SC>(state),
        }
    }
}
