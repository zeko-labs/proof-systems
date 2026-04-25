// src/poseidon.rs
use crate::sp1::fp::Fp;
use crate::sp1::params::{FULL_ROUNDS, MDS, RATE, ROUND_CONSTANTS, WIDTH};

#[derive(Clone)]
pub struct Sponge {
    state: [Fp; WIDTH],
}

impl Sponge {
    pub fn new() -> Self {
        Self {
            state: [Fp::ZERO; WIDTH],
        }
    }

    fn add_round_constants(&mut self, round: usize) {
        for i in 0..WIDTH {
            self.state[i] = self.state[i].add(ROUND_CONSTANTS[round][i]);
        }
    }

    fn sbox(&mut self) {
        for i in 0..WIDTH {
            self.state[i] = self.state[i].pow7();
        }
    }

    // MDS: 9 mul + 6 add — fully unrolled by the compiler at -O3
    fn mds(&mut self) {
        let s = self.state;
        for row in 0..WIDTH {
            self.state[row] =
                (0..WIDTH).fold(Fp::ZERO, |acc, col| acc.add(MDS[row][col].mul(s[col])));
        }
    }

    pub fn permute(&mut self) {
        for round in 0..FULL_ROUNDS {
            // 1. S-box
            for i in 0..WIDTH {
                self.state[i] = self.state[i].pow7();
            }

            // 2. MDS + add round constants in one pass (matches original)
            let s = self.state;
            for row in 0..WIDTH {
                self.state[row] =
                    (0..WIDTH).fold(Fp::ZERO, |acc, col| acc.add(MDS[row][col].mul(s[col])));
                self.state[row] = self.state[row].add(ROUND_CONSTANTS[round][row]);
            }
        }
    }

    pub fn absorb(&mut self, inputs: &[Fp]) {
        for chunk in inputs.chunks(RATE) {
            for (i, &x) in chunk.iter().enumerate() {
                self.state[i] = self.state[i].add(x);
            }
            self.permute();
        }
    }

    pub fn squeeze(&self) -> Fp {
        self.state[0]
    }

    // Convenience: hash a fixed pair (most common in Mina Merkle trees)
    pub fn hash_pair(left: Fp, right: Fp) -> Fp {
        let mut s = Self::new();
        s.absorb(&[left, right]);
        s.squeeze()
    }

    pub fn hash(inputs: &[Fp]) -> Fp {
        let mut s = Self::new();
        if inputs.is_empty() {
            s.permute(); // original behavior: one permutation on empty input
        } else {
            s.absorb(inputs);
        }
        s.squeeze()
    }
}
