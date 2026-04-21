use std::{fs, path::Path, path::PathBuf, sync::Arc};

use ark_serialize::CanonicalDeserialize;
use kimchi::{
    circuits::constraints::FeatureFlags,
    groupmap::GroupMap,
    linearization::expr_linearization,
    mina_curves::pasta::{Fp, Fq, Pallas, PallasParameters},
    proof::ProverProof,
    verifier_index::VerifierIndex,
};
use mina_poseidon::{
    constants::PlonkSpongeConstantsKimchi,
    sponge::{DefaultFqSponge, DefaultFrSponge},
};
use poly_commitment::{
    ipa::{endos, OpeningProof, SRS as IPASrs},
    SRS as SRSTrait,
};

const FULL_ROUNDS: usize = mina_poseidon::pasta::FULL_ROUNDS;

type SpongeParams = PlonkSpongeConstantsKimchi;
type EFqSponge = DefaultFqSponge<PallasParameters, SpongeParams, FULL_ROUNDS>;
type EFrSponge = DefaultFrSponge<Fq, SpongeParams, FULL_ROUNDS>;

type Opening = OpeningProof<Pallas, FULL_ROUNDS>;
type Srs = IPASrs<Pallas>;
type Index = VerifierIndex<FULL_ROUNDS, Pallas, Srs>;
type Proof = ProverProof<Pallas, Opening, FULL_ROUNDS>;

fn restore_verifier_index_runtime_fields(index: &mut Index) {
    let feature_flags = FeatureFlags {
        range_check0: index.range_check0_comm.is_some(),
        range_check1: index.range_check1_comm.is_some(),
        foreign_field_add: index.foreign_field_add_comm.is_some(),
        foreign_field_mul: index.foreign_field_mul_comm.is_some(),
        xor: index.xor_comm.is_some(),
        rot: index.rot_comm.is_some(),
        lookup_features: index
            .lookup_index
            .as_ref()
            .map(|li| li.lookup_info.features)
            .unwrap_or_default(),
    };

    let (linearization, powers_of_alpha) = expr_linearization(Some(&feature_flags), true);
    index.linearization = linearization;
    index.powers_of_alpha = powers_of_alpha;
}

fn load_verify_fixture(
    verifier_index_path: &Path,
    proof_path: &Path,
    srs: Arc<Srs>,
    endo_q: Fq,
) -> (Index, Proof, Vec<Fq>) {
    let mut verifier_index = Index::from_file(srs, verifier_index_path, None, endo_q)
        .expect("failed to load verifier index");

    restore_verifier_index_runtime_fields(&mut verifier_index);

    let bytes = fs::read(proof_path).expect("failed to read proof payload");

    let (proof, public_input_bytes): (Proof, Vec<[u8; 32]>) =
        rmp_serde::from_slice(&bytes).expect("failed to deserialize proof payload");

    let public_input = public_input_bytes
        .into_iter()
        .map(|buf| Fq::deserialize_uncompressed(&buf[..]).expect("failed to deserialize Fq"))
        .collect();

    (verifier_index, proof, public_input)
}

#[test]
fn kimchi_proof() {
    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let verifier_index_path = tests_dir.join("kimchi_verify_index.bin");
    let proof_path = tests_dir.join("kimchi_verify_proof.bin");

    println!(
        "Loading proof and verifier index from fixtures in {}",
        tests_dir.display()
    );

    let srs = Arc::new(Srs::create(1 << 15));
    let endo_q = endos::<Pallas>().1;

    let (verifier_index, proof, public_input) =
        load_verify_fixture(&verifier_index_path, &proof_path, srs, endo_q);

    let group_map = GroupMap::<Fp>::setup();

    let res = kimchi::verifier::verify::<
        FULL_ROUNDS,
        Pallas,
        EFqSponge,
        EFrSponge,
        Opening,
    >(&group_map, &verifier_index, &proof, &public_input);

    assert!(res.is_ok(), "verify failed: {:?}", res);
}