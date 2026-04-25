// lib.rs
pub mod fp;
pub mod params;
pub mod poseidon;

pub use fp::Fp;
use mina_curves::pasta::Fp as FpMina;
use mina_poseidon::pasta::fp_kimchi;
use mina_poseidon::{
    constants::PlonkSpongeConstantsKimchi,
    pasta::FULL_ROUNDS,
    poseidon::{ArithmeticSponge as Poseidon, Sponge as MinaSponge},
};

pub fn poseidon_hash(input: &[FpMina]) -> FpMina {
    let mut hash = Poseidon::<FpMina, PlonkSpongeConstantsKimchi, FULL_ROUNDS>::new(
        fp_kimchi::static_params(),
    );
    hash.absorb(input);
    hash.squeeze()
}
