#[test]
fn kimchi_proof() {
    use ark_serialize::CanonicalDeserialize;
    use kimchi::{groupmap::GroupMap, verifier::verify, verifier_index::VerifierIndex};
    use mina_curves::pasta::{Fp, Fq, Pallas};
    use mina_poseidon::sponge::{DefaultFqSponge, DefaultFrSponge};
    use poly_commitment::ipa::{OpeningProof, SRS};
    use std::{fs, path::PathBuf, sync::Arc};

    type SpongeParams = mina_poseidon::constants::PlonkSpongeConstantsKimchi;
    type EFqSponge = DefaultFqSponge<mina_curves::pasta::PallasParameters, SpongeParams, 55>;
    type EFrSponge = DefaultFrSponge<Fq, SpongeParams, 55>;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    // Load SRS — sérialisé comme (g_bytes, h_bytes) via bincode
    let srs_bytes = fs::read(dir.join("srs.bin")).expect("missing srs.bin");
    let (g_bytes, h_bytes): (Vec<u8>, Vec<u8>) = bincode::deserialize(&srs_bytes).unwrap();
    let g: Vec<Pallas> = CanonicalDeserialize::deserialize_uncompressed(&g_bytes[..]).unwrap();
    let h: Pallas = CanonicalDeserialize::deserialize_uncompressed(&h_bytes[..]).unwrap();
    let mut srs = SRS::<Pallas>::default();
    srs.g = g;
    srs.h = h;
    let srs = Arc::new(srs);

    // Load verifier index
    let vi_path = dir.join("verifier_index.bin");
    let (_endo_q, endo_r) = poly_commitment::ipa::endos::<Pallas>();

    let mut vi: VerifierIndex<55, Pallas, SRS<Pallas>> =
        VerifierIndex::from_file(srs.clone(), &vi_path, None, endo_r).unwrap();
    vi.srs = srs;

    // Load proof + public inputs — sérialisés via rmp_serde
    let proof_bytes = fs::read(dir.join("proof.bin")).unwrap();
    let (proof, pi_bytes): (
        kimchi::proof::ProverProof<Pallas, OpeningProof<Pallas, 55>, 55>,
        Vec<[u8; 32]>,
    ) = rmp_serde::from_slice(&proof_bytes).unwrap();

    let public_inputs: Vec<Fq> = pi_bytes
        .iter()
        .map(|b| Fq::deserialize_uncompressed(&b[..]).unwrap())
        .collect();

    let group_map = GroupMap::<Fp>::setup();

    let result = verify::<55, Pallas, EFqSponge, EFrSponge, OpeningProof<Pallas, 55>>(
        &group_map,
        &vi,
        &proof,
        &public_inputs,
    );

    assert!(result.is_ok(), "verify failed: {:?}", result);
}
