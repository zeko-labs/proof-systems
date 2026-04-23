use std::collections::HashMap;

use ark_poly::domain::EvaluationDomain;
use mina_curves::pasta::Vesta;
use poly_commitment::{hash_map_cache::HashMapCache, SRS};
use rkyv::{util::AlignedVec, Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
// ---------------------------------------------------------------------------
// rkyv structs — used only for static SRS file embedding (include_bytes!)
// ---------------------------------------------------------------------------

#[derive(Archive, RkyvSerialize, RkyvDeserialize, Clone, Debug)]
#[rkyv(derive(Debug))]
pub struct RkyvPoint {
    pub x: [u8; 32],
    pub y: [u8; 32],
    pub infinity: bool,
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize, Clone, Debug)]
#[rkyv(derive(Debug))]
pub struct RkyvPolyComm {
    pub chunks: Vec<RkyvPoint>,
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize, Clone, Debug)]
#[rkyv(derive(Debug))]
pub struct RkyvSRS {
    pub g: Vec<RkyvPoint>,
    pub h: RkyvPoint,
    pub domain_size: usize,
    pub lagrange_bases: Vec<RkyvPolyComm>,
}

#[test]
fn kimchi_proof() {
    use ark_serialize::CanonicalDeserialize;
    use kimchi::{
        circuits::constraints::FeatureFlags, groupmap::GroupMap, linearization::expr_linearization,
        verifier::verify, verifier_index::VerifierIndex,
    };
    use mina_curves::pasta::{Fp, Fq, Pallas, PallasParameters};
    use mina_poseidon::sponge::{DefaultFqSponge, DefaultFrSponge};
    use poly_commitment::{
        hash_map_cache::HashMapCache,
        ipa::{endos, OpeningProof, SRS},
    };
    use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};

    type SpongeParams = mina_poseidon::constants::PlonkSpongeConstantsKimchi;
    type EFqSponge = DefaultFqSponge<PallasParameters, SpongeParams, 55>;
    type EFrSponge = DefaultFrSponge<Fq, SpongeParams, 55>;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    // ------------------------------------------------------------------
    // SRS — exactement comme main.rs dans sp1-verifier
    // ------------------------------------------------------------------
    let srs_bytes = fs::read(dir.join("srs_rkyv.bin")).expect("missing srs_rkyv.bin");
    let mut aligned = rkyv::util::AlignedVec::<16>::new();
    aligned.extend_from_slice(&srs_bytes);
    let archived = unsafe { rkyv::access_unchecked::<ArchivedRkyvSRS>(&aligned) };

    let g: Vec<Pallas> = archived
        .g
        .iter()
        .map(|p| {
            if p.infinity {
                return Pallas::default();
            }
            let x = mina_curves::pasta::Fp::deserialize_uncompressed(&p.x[..]).unwrap();
            let y = mina_curves::pasta::Fp::deserialize_uncompressed(&p.y[..]).unwrap();
            Pallas::new_unchecked(x, y)
        })
        .collect();

    let h: Pallas = {
        let p = &archived.h;
        if p.infinity {
            Pallas::default()
        } else {
            let x = mina_curves::pasta::Fp::deserialize_uncompressed(&p.x[..]).unwrap();
            let y = mina_curves::pasta::Fp::deserialize_uncompressed(&p.y[..]).unwrap();
            Pallas::new_unchecked(x, y)
        }
    };

    let lagrange_bases: Vec<poly_commitment::PolyComm<Pallas>> = archived
        .lagrange_bases
        .iter()
        .map(|comm| poly_commitment::PolyComm {
            chunks: comm
                .chunks
                .iter()
                .map(|p| {
                    if p.infinity {
                        return Pallas::default();
                    }
                    let x = mina_curves::pasta::Fp::deserialize_uncompressed(&p.x[..]).unwrap();
                    let y = mina_curves::pasta::Fp::deserialize_uncompressed(&p.y[..]).unwrap();
                    Pallas::new_unchecked(x, y)
                })
                .collect(),
        })
        .collect();

    let domain_size = archived.domain_size.to_native();
    let mut map = HashMap::new();
    map.insert(domain_size.try_into().unwrap(), lagrange_bases);

    let srs = Arc::new(SRS::<Pallas> {
        g,
        h,
        lagrange_bases: HashMapCache::new_from_hashmap(map),
    });

    // ------------------------------------------------------------------
    // VerifierIndex — exactement comme main.rs
    // ------------------------------------------------------------------
    let vi_bytes = fs::read(dir.join("verifier_index_bincode.bin")).unwrap();
    let mut vi: VerifierIndex<55, Pallas, SRS<Pallas>> = bincode::deserialize(&vi_bytes).unwrap();
    vi.srs = srs;

    let feature_flags = FeatureFlags::default();
    let (linearization, powers_of_alpha) = expr_linearization(Some(&feature_flags), true);
    let (_endo_q, endo_r) = endos::<Vesta>();
    vi.linearization = linearization;
    vi.powers_of_alpha = powers_of_alpha;
    vi.endo = _endo_q;

    // ------------------------------------------------------------------
    // Proof + public inputs — depuis /tmp/kimchi_fixture
    // ------------------------------------------------------------------
    let proof_bytes = fs::read(dir.join("proof.bin")).unwrap();
    let (proof, pi_bytes): (
        kimchi::proof::ProverProof<Pallas, OpeningProof<Pallas, 55>, 55>,
        Vec<[u8; 32]>,
    ) = rmp_serde::from_slice(&proof_bytes).unwrap();

    let public_inputs: Vec<Fq> = pi_bytes
        .iter()
        .map(|b| Fq::deserialize_uncompressed(&b[..]).unwrap())
        .collect();

    println!("vi.domain.size = {}", vi.domain.size());
    println!("srs.g.len = {}", vi.srs.g.len());
    println!("public_inputs.len = {}", public_inputs.len());
    println!("vi.endo = {:?}", vi.endo);

    // ------------------------------------------------------------------
    // Verify
    // ------------------------------------------------------------------
    let group_map = GroupMap::<Fp>::setup();
    let result = verify::<55, Pallas, EFqSponge, EFrSponge, OpeningProof<Pallas, 55>>(
        &group_map,
        &vi,
        &proof,
        &public_inputs,
    );

    match result {
        Ok(_) => {}
        Err(e) => panic!("verify failed: {:?}", e),
    }
}
