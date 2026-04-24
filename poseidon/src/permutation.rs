//! The permutation module contains the function implementing the permutation
//! used in Poseidon.

extern crate alloc;

use crate::{
    constants::SpongeConstants,
    poseidon::{sbox, ArithmeticSpongeParams},
};
use ark_ff::Field;

use core::sync::atomic::{AtomicU64, Ordering};

const MDS_WIDTH: usize = 3;

fn apply_mds_matrix<F: Field, SC: SpongeConstants>(
    mds: [[F; MDS_WIDTH]; MDS_WIDTH],
    state: &mut [F],
) {
    // Fast path for the special MDS shape.
    if !SC::PERM_FULL_MDS {
        let s0 = state[0];
        let s1 = state[1];
        let s2 = state[2];

        state[0] = s0 + s2;
        state[1] = s0 + s1;
        state[2] = s1 + s2;
        return;
    }

    let mut new_state = [F::zero(); MDS_WIDTH];

    for (new_state, mds) in new_state.iter_mut().zip(mds.iter()) {
        *new_state = mds
            .iter()
            .copied()
            .zip(state.iter())
            .map(|(md, state)| md * state)
            .sum();
    }

    new_state
        .into_iter()
        .zip(state.iter_mut())
        .for_each(|(new_s, s)| {
            *s = new_s;
        });
}

/// Apply a full round of the permutation.
/// A full round is composed of the following steps:
/// - Apply the S-box to each element of the state.
/// - Apply the MDS matrix to the state.
/// - Add the round constants to the state.
///
/// The function has side-effect and the parameter state is modified.
pub(crate) fn full_round<F: Field, SC: SpongeConstants, const FULL_ROUNDS: usize>(
    params: &ArithmeticSpongeParams<F, FULL_ROUNDS>,
    state: &mut [F],
    r: usize,
) {
    state.iter_mut().for_each(|s| {
        *s = sbox::<F, SC>(*s);
    });
    let mds = params.mds;

    apply_mds_matrix::<F, SC>(mds, state);

    for (i, x) in params.round_constants[r].iter().enumerate() {
        state[i].add_assign(x);
    }
}

pub fn half_rounds<F: Field, SC: SpongeConstants, const FULL_ROUNDS: usize>(
    params: &ArithmeticSpongeParams<F, FULL_ROUNDS>,
    state: &mut [F],
) {
    for r in 0..SC::PERM_HALF_ROUNDS_FULL {
        for (i, x) in params.round_constants[r].iter().enumerate() {
            state[i].add_assign(x);
        }

        for state_i in state.iter_mut() {
            *state_i = sbox::<F, SC>(*state_i);
        }

        apply_mds_matrix::<F, SC>(params.mds, state);
    }

    for r in 0..SC::PERM_ROUNDS_PARTIAL {
        for (i, x) in params.round_constants[SC::PERM_HALF_ROUNDS_FULL + r]
            .iter()
            .enumerate()
        {
            state[i].add_assign(x);
        }
        state[0] = sbox::<F, SC>(state[0]);

        apply_mds_matrix::<F, SC>(params.mds, state);
    }

    for r in 0..SC::PERM_HALF_ROUNDS_FULL {
        for (i, x) in params.round_constants
            [SC::PERM_HALF_ROUNDS_FULL + SC::PERM_ROUNDS_PARTIAL + r]
            .iter()
            .enumerate()
        {
            state[i].add_assign(x);
        }

        for state_i in state.iter_mut() {
            *state_i = sbox::<F, SC>(*state_i);
        }

        apply_mds_matrix::<F, SC>(params.mds, state);
    }
}

pub fn poseidon_block_cipher<F: Field, SC: SpongeConstants, const FULL_ROUNDS: usize>(
    params: &ArithmeticSpongeParams<F, FULL_ROUNDS>,
    state: &mut [F],
) {
    #[cfg(target_os = "zkvm")]
    {
        sp1::permute_sp1::<F, SC, FULL_ROUNDS>(params, state);
        return;
    }

    if SC::PERM_HALF_ROUNDS_FULL == 0 {
        if SC::PERM_INITIAL_ARK {
            // Keep the previous invariant.
            assert!(params.round_constants[0].len() <= state.len());

            state
                .iter_mut()
                .zip(params.round_constants[0].iter())
                .for_each(|(s, x)| {
                    s.add_assign(x);
                });

            for r in 0..SC::PERM_ROUNDS_FULL {
                full_round::<_, SC, FULL_ROUNDS>(params, state, r + 1);
            }
        } else {
            for r in 0..SC::PERM_ROUNDS_FULL {
                full_round::<_, SC, FULL_ROUNDS>(params, state, r);
            }
        }
    } else {
        half_rounds::<_, SC, FULL_ROUNDS>(params, state);
    }
}

#[cfg(target_os = "zkvm")]
mod sp1 {
    use crate::constants::SpongeConstants;
    use crate::poseidon::ArithmeticSpongeParams;
    use ark_ff::PrimeField;
    use core::array;
    use bytemuck;

    const ZERO_LIMBS: [u64; 4] = [0u64; 4];

    #[derive(Clone, Copy)]
    struct Fp {
        limbs: [u64; 4],
        modulus: *const [u64; 4],
    }

    fn modulus_limbs<F: PrimeField>() -> [u64; 4] {
        let m = F::MODULUS;
        unsafe { *(m.as_ref().as_ptr() as *const [u64; 4]) }
    }

    #[inline(always)]
    fn from_ark<F: PrimeField>(x: F, modulus: &[u64; 4]) -> Fp {
        let b: [u64; 4] = unsafe { *(x.into_bigint().as_ref().as_ptr() as *const [u64; 4]) };
        Fp { limbs: b, modulus: modulus as *const [u64; 4] }
    }

    #[inline(always)]
    fn to_ark<F: PrimeField>(fp: Fp) -> F {
        use ark_ff::BigInteger;
        let mut bi = F::BigInt::from(0u64);
        bi.as_mut().copy_from_slice(&fp.limbs);
        F::from_bigint(bi).unwrap()
    }

    #[inline(always)]
    fn mul(a: Fp, b: Fp) -> Fp {
        let mut result = [0u64; 4];
        unsafe {
            sp1_lib::sys_bigint(&mut result, 0, &a.limbs, &b.limbs, &*a.modulus);
        }
        Fp { limbs: result, modulus: a.modulus }
    }

    #[inline(always)]
    fn add(a: Fp, b: Fp) -> Fp {
        // Addition manuelle avec réduction conditionnelle
        let mut carry = 0u64;
        let mut result = [0u64; 4];
        let m = unsafe { &*a.modulus };
        for i in 0..4 {
            let (s1, c1) = a.limbs[i].overflowing_add(b.limbs[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }
        // Réduction conditionnelle
        let need_reduce = carry > 0 || {
            let mut ge = false;
            let mut eq = true;
            for i in (0..4).rev() {
                if !eq { break; }
                if result[i] > m[i] { ge = true; eq = false; }
                else if result[i] < m[i] { eq = false; }
            }
            ge || eq
        };
        if need_reduce {
            let mut borrow = 0u64;
            for i in 0..4 {
                let (d1, b1) = result[i].overflowing_sub(m[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                result[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
            }
        }
        Fp { limbs: result, modulus: a.modulus }
    }

    #[inline(always)]
    fn pow7(x: Fp) -> Fp {
        let x2 = mul(x, x);
        let x4 = mul(x2, x2);
        let x6 = mul(x4, x2);
        mul(x6, x)
    }

    #[inline(always)]
    fn full_round_sp1<F: PrimeField>(
        state: &mut [Fp; 3],
        mds: &[[Fp; 3]; 3],
        rc: &[F; 3],
    ) {
        // S-box
        for i in 0..3 { state[i] = pow7(state[i]); }

        // MDS
        let tmp = *state;
        let modulus = state[0].modulus;
        for row in 0..3 {
            state[row] = (0..3).fold(
                Fp { limbs: ZERO_LIMBS, modulus },
                |acc, col| add(acc, mul(mds[row][col], tmp[col]))
            );
        }

        // Round constants
        for i in 0..3 {
            let rc_fp = from_ark(rc[i], unsafe { &*modulus });
            state[i] = add(state[i], rc_fp);
        }
    }

    pub fn permute_sp1<F: PrimeField, SC: SpongeConstants, const FULL_ROUNDS: usize>(
        params: &ArithmeticSpongeParams<F, FULL_ROUNDS>,
        state: &mut [F],
    ) {
        let modulus: [u64; 4] = modulus_limbs::<F>();

        // Convertit state → Fp
        let mut s: [Fp; 3] = array::from_fn(|i| from_ark(state[i], &modulus));

        // Préconvertit MDS
        let mds: [[Fp; 3]; 3] = array::from_fn(|row| {
            array::from_fn(|col| from_ark(params.mds[row][col], &modulus))
        });

        if SC::PERM_INITIAL_ARK {
            for i in 0..3 {
                let rc = from_ark(params.round_constants[0][i], &modulus);
                s[i] = add(s[i], rc);
            }
            for r in 0..SC::PERM_ROUNDS_FULL {
                full_round_sp1(&mut s, &mds, &params.round_constants[r + 1]);
            }
        } else {
            for r in 0..SC::PERM_ROUNDS_FULL {
                full_round_sp1(&mut s, &mds, &params.round_constants[r]);
            }
        }

        // Reconvertit → ark-ff
        for i in 0..3 {
            state[i] = to_ark(s[i]);
        }
    }
}