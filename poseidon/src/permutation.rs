//! The permutation module contains the function implementing the permutation
//! used in Poseidon.

extern crate alloc;

use crate::{
    constants::SpongeConstants,
    poseidon::{sbox, ArithmeticSpongeParams},
};
use ark_ff::{Field, PrimeField};

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
        // Pallas Fp modulus — works for both Pallas and Vesta
        // detected at runtime via field size if needed
        let char = F::characteristic();
        let modulus: [u64; 4] = [
            char.get(0).copied().unwrap_or(0),
            char.get(1).copied().unwrap_or(0),
            char.get(2).copied().unwrap_or(0),
            char.get(3).copied().unwrap_or(0),
        ];
        sp1::permute_sp1::<F, SC, FULL_ROUNDS>(params, state, modulus);
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
    use super::*;
    use crate::constants::SpongeConstants;
    use crate::pasta::fp_sp1::{MDS as SP1_MDS, ROUND_CONSTANTS as SP1_RC};
    use crate::poseidon::ArithmeticSpongeParams;
    use ark_ff::Field;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use core::array;

    const ZERO_LIMBS: [u64; 4] = [0u64; 4];

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    struct Sp1Fp([u64; 4]);

    #[inline(always)]
    fn from_ark<F: CanonicalSerialize>(x: F) -> Sp1Fp {
        let mut buf = [0u8; 32];
        x.serialize_uncompressed(&mut buf[..]).unwrap();
        let limbs: [u64; 4] = bytemuck::cast(buf);
        Sp1Fp(limbs)
    }

    #[inline(always)]
    fn to_ark<F: CanonicalDeserialize>(fp: Sp1Fp) -> F {
        let buf: [u8; 32] = bytemuck::cast(fp.0);
        F::deserialize_uncompressed(&buf[..]).unwrap()
    }

    #[inline(always)]
    fn add(a: Sp1Fp, b: Sp1Fp, modulus: [u64; 4]) -> Sp1Fp {
        let mut carry = 0u64;
        let mut result = [0u64; 4];

        for i in 0..4 {
            let (s1, c1) = a.0[i].overflowing_add(b.0[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }

        let need_reduce = carry > 0 || {
            let mut ge = false;
            let mut eq = true;
            for i in (0..4).rev() {
                if !eq {
                    break;
                }
                if result[i] > modulus[i] {
                    ge = true;
                    eq = false;
                } else if result[i] < modulus[i] {
                    eq = false;
                }
            }
            ge || eq
        };

        if need_reduce {
            let mut borrow = 0u64;
            for i in 0..4 {
                let (d1, b1) = result[i].overflowing_sub(modulus[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                result[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
            }
        }

        Sp1Fp(result)
    }

    #[inline(always)]
    fn mul(a: Sp1Fp, b: Sp1Fp, modulus: [u64; 4]) -> Sp1Fp {
        let mut result = [0u64; 4];
        #[allow(unsafe_code)]
        unsafe {
            sp1_lib::sys_bigint(&mut result, 0, &a.0, &b.0, &modulus);
        }
        Sp1Fp(result)
    }

    #[inline(always)]
    fn pow7(x: Sp1Fp, modulus: [u64; 4]) -> Sp1Fp {
        let x2 = mul(x, x, modulus);
        let x4 = mul(x2, x2, modulus);
        let x6 = mul(x4, x2, modulus);
        mul(x6, x, modulus)
    }

    #[inline(always)]
    fn full_round_sp1(
        state: &mut [Sp1Fp; 3],
        mds: &[[[u64; 4]; 3]; 3],
        rc: &[[u64; 4]; 3],
        modulus: [u64; 4],
    ) {
        // Apply the x^7 S-box to each element.
        for i in 0..3 {
            state[i] = pow7(state[i], modulus);
        }

        // Apply the MDS matrix.
        let tmp = *state;
        for row in 0..3 {
            let mut acc = Sp1Fp(ZERO_LIMBS);
            for col in 0..3 {
                acc = add(acc, mul(Sp1Fp(mds[row][col]), tmp[col], modulus), modulus);
            }
            state[row] = acc;
        }

        // Add round constants.
        for i in 0..3 {
            state[i] = add(state[i], Sp1Fp(rc[i]), modulus);
        }
    }

    pub fn permute_sp1<
        F: Field + CanonicalSerialize + CanonicalDeserialize + Copy,
        SC: SpongeConstants,
        const FULL_ROUNDS: usize,
    >(
        _params: &ArithmeticSpongeParams<F, FULL_ROUNDS>,
        state: &mut [F],
        modulus: [u64; 4],
    ) {
        // Convert ark field elements into the SP1-friendly representation.
        let mut s: [Sp1Fp; 3] = array::from_fn(|i| from_ark(state[i]));

        let mds: [[[u64; 4]; 3]; 3] = SP1_MDS;
        let rc: [[[u64; 4]; 3]; 55] = SP1_RC;

        if SC::PERM_INITIAL_ARK {
            for i in 0..3 {
                s[i] = add(s[i], Sp1Fp(rc[0][i]), modulus);
            }
            for r in 0..SC::PERM_ROUNDS_FULL {
                full_round_sp1(&mut s, &mds, &rc[r + 1], modulus);
            }
        } else {
            for r in 0..SC::PERM_ROUNDS_FULL {
                full_round_sp1(&mut s, &mds, &rc[r], modulus);
            }
        }

        // Convert back into ark field elements.
        for i in 0..3 {
            state[i] = to_ark(s[i]);
        }
    }
}
