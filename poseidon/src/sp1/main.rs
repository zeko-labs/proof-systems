// src/main.rs
use crate::{Fp, Sponge};

fn main() {
    // Hash a single value
    let mut sponge = Sponge::new();
    sponge.absorb(&[Fp::from(12u64)]);
    println!(
        "Poseidon hash of [12]: {:?}",
        sponge.squeeze().to_be_bytes()
    );

    // Hash multiple values
    let mut sponge = Sponge::new();
    sponge.absorb(&[Fp::from(3412u64), Fp::from(548748548u64)]);
    println!(
        "Poseidon hash of [3412, 548748548]: {:?}",
        sponge.squeeze().to_be_bytes()
    );
}
