use crate::{
    constants::SpongeConstants,
    poseidon::{sbox, ArithmeticSpongeParams},
};
use ark_ff::{Field, PrimeField};

const MDS_WIDTH: usize = 3;

const PALLAS_BASE_MODULUS: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
];

const VESTA_BASE_MODULUS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
];

#[inline(always)]
fn detect_pasta_modulus<F: PrimeField>() -> Option<[u64; 4]> {
    let ch = F::characteristic();
    let modulus = [
        ch.get(0).copied().unwrap_or(0),
        ch.get(1).copied().unwrap_or(0),
        ch.get(2).copied().unwrap_or(0),
        ch.get(3).copied().unwrap_or(0),
    ];

    match modulus {
        PALLAS_BASE_MODULUS => Some(PALLAS_BASE_MODULUS),
        VESTA_BASE_MODULUS => Some(VESTA_BASE_MODULUS),
        _ => None,
    }
}

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

    apply_mds_matrix::<F, SC>(params.mds, state);

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

pub fn poseidon_block_cipher<F: PrimeField, SC: SpongeConstants, const FULL_ROUNDS: usize>(
    params: &ArithmeticSpongeParams<F, FULL_ROUNDS>,
    state: &mut [F],
) {
    #[cfg(target_os = "zkvm")]
    {
        if let Some(modulus) = detect_pasta_modulus::<F>() {
            sp1::permute_sp1::<F, SC, FULL_ROUNDS>(params, state, modulus);
            return;
        }
    }

    if SC::PERM_HALF_ROUNDS_FULL == 0 {
        if SC::PERM_INITIAL_ARK {
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
    use ark_ff::PrimeField;
    use core::array;

    const ZERO_LIMBS: [u64; 4] = [0u64; 4];

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    struct Sp1Fp([u64; 4]);

    #[inline(always)]
    fn from_ark<F: PrimeField>(x: F) -> Sp1Fp {
        let bigint = x.into_bigint();
        let limbs = bigint.as_ref();
        debug_assert!(limbs.len() == 4);
        Sp1Fp([limbs[0], limbs[1], limbs[2], limbs[3]])
    }

    #[inline(always)]
    fn to_ark<F: PrimeField>(x: Sp1Fp) -> F {
        let bytes: [u8; 32] = bytemuck::cast(x.0);
        F::from_le_bytes_mod_order(&bytes)
    }

    #[inline(always)]
    fn add(a: Sp1Fp, b: Sp1Fp, modulus: [u64; 4]) -> Sp1Fp {
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
    fn mul(a: Sp1Fp, b: Sp1Fp, modulus: [u64; 4]) -> Sp1Fp {
        let mut out = [0u64; 4];
        #[allow(unsafe_code)]
        unsafe {
            sp1_lib::sys_bigint(&mut out, 0, &a.0, &b.0, &modulus);
        }
        Sp1Fp(out)
    }

    #[inline(always)]
    fn pow7(x: Sp1Fp, modulus: [u64; 4]) -> Sp1Fp {
        let x2 = mul(x, x, modulus);
        let x4 = mul(x2, x2, modulus);
        let x6 = mul(x4, x2, modulus);
        mul(x6, x, modulus)
    }

    #[inline(always)]
    fn apply_mds_matrix_sp1<SC: SpongeConstants>(
        mds: &[[Sp1Fp; 3]; 3],
        state: &mut [Sp1Fp; 3],
        modulus: [u64; 4],
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
                mul(mds[0][0], tmp[0], modulus),
                mul(mds[0][1], tmp[1], modulus),
                modulus,
            ),
            mul(mds[0][2], tmp[2], modulus),
            modulus,
        );

        state[1] = add(
            add(
                mul(mds[1][0], tmp[0], modulus),
                mul(mds[1][1], tmp[1], modulus),
                modulus,
            ),
            mul(mds[1][2], tmp[2], modulus),
            modulus,
        );

        state[2] = add(
            add(
                mul(mds[2][0], tmp[0], modulus),
                mul(mds[2][1], tmp[1], modulus),
                modulus,
            ),
            mul(mds[2][2], tmp[2], modulus),
            modulus,
        );
    }

    #[inline(always)]
    fn full_round_sp1<SC: SpongeConstants>(
        state: &mut [Sp1Fp; 3],
        mds: &[[Sp1Fp; 3]; 3],
        rc: &[Sp1Fp; 3],
        modulus: [u64; 4],
    ) {
        state[0] = pow7(state[0], modulus);
        state[1] = pow7(state[1], modulus);
        state[2] = pow7(state[2], modulus);

        apply_mds_matrix_sp1::<SC>(mds, state, modulus);

        state[0] = add(state[0], rc[0], modulus);
        state[1] = add(state[1], rc[1], modulus);
        state[2] = add(state[2], rc[2], modulus);
    }

    #[inline(always)]
    fn half_rounds_sp1<SC: SpongeConstants, const FULL_ROUNDS: usize>(
        mds: &[[Sp1Fp; 3]; 3],
        rc: &[[Sp1Fp; 3]; FULL_ROUNDS],
        state: &mut [Sp1Fp; 3],
        modulus: [u64; 4],
    ) {
        for r in 0..SC::PERM_HALF_ROUNDS_FULL {
            state[0] = add(state[0], rc[r][0], modulus);
            state[1] = add(state[1], rc[r][1], modulus);
            state[2] = add(state[2], rc[r][2], modulus);

            state[0] = pow7(state[0], modulus);
            state[1] = pow7(state[1], modulus);
            state[2] = pow7(state[2], modulus);

            apply_mds_matrix_sp1::<SC>(mds, state, modulus);
        }

        for r in 0..SC::PERM_ROUNDS_PARTIAL {
            let rr = SC::PERM_HALF_ROUNDS_FULL + r;

            state[0] = add(state[0], rc[rr][0], modulus);
            state[1] = add(state[1], rc[rr][1], modulus);
            state[2] = add(state[2], rc[rr][2], modulus);

            state[0] = pow7(state[0], modulus);

            apply_mds_matrix_sp1::<SC>(mds, state, modulus);
        }

        for r in 0..SC::PERM_HALF_ROUNDS_FULL {
            let rr = SC::PERM_HALF_ROUNDS_FULL + SC::PERM_ROUNDS_PARTIAL + r;

            state[0] = add(state[0], rc[rr][0], modulus);
            state[1] = add(state[1], rc[rr][1], modulus);
            state[2] = add(state[2], rc[rr][2], modulus);

            state[0] = pow7(state[0], modulus);
            state[1] = pow7(state[1], modulus);
            state[2] = pow7(state[2], modulus);

            apply_mds_matrix_sp1::<SC>(mds, state, modulus);
        }
    }

    pub fn permute_sp1<F: PrimeField + Copy, SC: SpongeConstants, const FULL_ROUNDS: usize>(
        params: &ArithmeticSpongeParams<F, FULL_ROUNDS>,
        state: &mut [F],
        modulus: [u64; 4],
    ) {
        let mut s: [Sp1Fp; 3] = array::from_fn(|i| from_ark(state[i]));
        let mds: [[Sp1Fp; 3]; 3] =
            array::from_fn(|row| array::from_fn(|col| from_ark(params.mds[row][col])));
        let rc: [[Sp1Fp; 3]; FULL_ROUNDS] =
            array::from_fn(|round| array::from_fn(|i| from_ark(params.round_constants[round][i])));

        if SC::PERM_HALF_ROUNDS_FULL == 0 {
            if SC::PERM_INITIAL_ARK {
                s[0] = add(s[0], rc[0][0], modulus);
                s[1] = add(s[1], rc[0][1], modulus);
                s[2] = add(s[2], rc[0][2], modulus);

                for r in 0..SC::PERM_ROUNDS_FULL {
                    full_round_sp1::<SC>(&mut s, &mds, &rc[r + 1], modulus);
                }
            } else {
                for r in 0..SC::PERM_ROUNDS_FULL {
                    full_round_sp1::<SC>(&mut s, &mds, &rc[r], modulus);
                }
            }
        } else {
            half_rounds_sp1::<SC, FULL_ROUNDS>(&mds, &rc, &mut s, modulus);
        }

        state[0] = to_ark(s[0]);
        state[1] = to_ark(s[1]);
        state[2] = to_ark(s[2]);
    }
}
