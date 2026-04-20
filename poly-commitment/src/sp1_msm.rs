//! SP1-optimized MSM for Pallas using crypto-bigint + sys_bigint precompile.
//! Pallas: y² = x³ + 5, BaseField = Fp, ScalarField = Fq

use crypto_bigint::{Encoding, NonZero, U256, U512};

// ---------------------------------------------------------------------------
// Fp — Pallas base field (coordonnées des points)
// Modulus = 28948022309329048855892746252171976963363056481941560715954676764349967630337
// ---------------------------------------------------------------------------
const FP_MODULUS: U256 =
    U256::from_be_hex("40000000000000000000000000000000224698fc094cf91b992d30ed00000001");
const FP_MODULUS_LIMBS: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
];

// ---------------------------------------------------------------------------
// Fq — Pallas scalar field (scalaires du MSM)
// Modulus = 28948022309329048855892746252171976963363056481941647379679742748393362948097
// ---------------------------------------------------------------------------
const FQ_MODULUS_LIMBS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
];

// ---------------------------------------------------------------------------
// Fp field element — coordonnées affines/projectives
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp(U256);

impl Fp {
    pub const ZERO: Self = Fp(U256::ZERO);
    pub const ONE: Self = Fp(U256::ONE);

    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        Fp(self.0.add_mod(&rhs.0, &FP_MODULUS))
    }

    #[inline(always)]
    pub fn sub(self, rhs: Self) -> Self {
        Fp(self.0.sub_mod(&rhs.0, &FP_MODULUS))
    }

    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        #[cfg(target_os = "zkvm")]
        {
            println!("SP1 MSM on zkVM: using sys_bigint precompile");
            let lhs: [u64; 4] = bytemuck::cast(self.0.to_le_bytes());
            let rhs_l: [u64; 4] = bytemuck::cast(rhs.0.to_le_bytes());
            let mut result = [0u64; 4];
            #[allow(unsafe_code)]
            unsafe {
                sp1_lib::sys_bigint(
                    &mut result as *mut [u64; 4],
                    0,
                    &lhs as *const [u64; 4],
                    &rhs_l as *const [u64; 4],
                    &FP_MODULUS_LIMBS as *const [u64; 4],
                );
            }
            return Fp(U256::from_le_bytes(bytemuck::cast(result)));
        }
        #[cfg(not(target_os = "zkvm"))]
        {
            let (lo, hi) = self.0.mul_wide(&rhs.0);
            let wide = U512::from((lo, hi));
            let m512 = U512::from((FP_MODULUS, U256::ZERO));
            let (_, rem) = wide.div_rem(&NonZero::from_uint(m512));
            Fp(U256::from_le_bytes(
                rem.to_le_bytes()[..32].try_into().unwrap(),
            ))
        }
    }

    #[inline(always)]
    pub fn neg(self) -> Self {
        if self.0 == U256::ZERO {
            self
        } else {
            Fp(FP_MODULUS.wrapping_sub(&self.0))
        }
    }

    #[inline(always)]
    pub fn square(self) -> Self {
        self.mul(self)
    }

    pub fn inverse(self) -> Option<Self> {
        if self.0 == U256::ZERO {
            return None;
        }
        let exp = FP_MODULUS.wrapping_sub(&U256::from(2u64));
        let mut result = Self::ONE;
        let mut base = self;
        for i in 0..256usize {
            let byte = exp.to_le_bytes()[i / 8];
            if (byte >> (i % 8)) & 1 == 1 {
                result = result.mul(base);
            }
            base = base.square();
        }
        Some(result)
    }

    pub fn from_le_bytes(b: &[u8; 32]) -> Self {
        Fp(U256::from_le_bytes(*b))
    }

    pub fn to_le_bytes(self) -> [u8; 32] {
        self.0.to_le_bytes()
    }
}

// ---------------------------------------------------------------------------
// Pallas point en coordonnées projectives (X:Y:Z)
// Courbe : y² = x³ + 5 sur Fp  (COEFF_A=0, COEFF_B=5)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct PallasPoint {
    x: Fp,
    y: Fp,
    z: Fp,
}

impl PallasPoint {
    const INFINITY: Self = PallasPoint {
        x: Fp::ZERO,
        y: Fp::ONE,
        z: Fp::ZERO,
    };

    #[inline]
    fn from_affine(x: Fp, y: Fp) -> Self {
        PallasPoint { x, y, z: Fp::ONE }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.z.0 == U256::ZERO
    }

    #[inline]
    fn double(self) -> Self {
        if self.is_zero() {
            return self;
        }

        if self.y == Fp::ZERO {
            return Self::INFINITY;
        }

        let two = Fp(U256::from(2u64));
        let three = Fp(U256::from(3u64));

        // lambda = (3 * x^2) / (2 * y) for y^2 = x^3 + 5
        let numerator = three.mul(self.x.square());
        let denominator = two.mul(self.y);

        let Some(den_inv) = denominator.inverse() else {
            return Self::INFINITY;
        };

        let lambda = numerator.mul(den_inv);
        let x3 = lambda.square().sub(self.x).sub(self.x);
        let y3 = lambda.mul(self.x.sub(x3)).sub(self.y);

        Self::from_affine(x3, y3)
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }

        // Since we keep points in affine form when z != 0,
        // we can use the standard affine formulas safely.
        if self.x == rhs.x {
            // P + (-P) = O
            if self.y != rhs.y {
                return Self::INFINITY;
            }

            // P + P
            return self.double();
        }

        let dx = rhs.x.sub(self.x);
        let dy = rhs.y.sub(self.y);

        let Some(dx_inv) = dx.inverse() else {
            return Self::INFINITY;
        };

        let lambda = dy.mul(dx_inv);
        let x3 = lambda.square().sub(self.x).sub(rhs.x);
        let y3 = lambda.mul(self.x.sub(x3)).sub(self.y);

        Self::from_affine(x3, y3)
    }

    /// Scalar multiplication with a little-endian scalar.
    fn scalar_mul(self, scalar_bytes: &[u8; 32]) -> Self {
        let mut result = Self::INFINITY;
        let mut base = self;

        for byte in scalar_bytes.iter() {
            for bit in 0..8u32 {
                if (byte >> bit) & 1 == 1 {
                    result = result.add(base);
                }
                base = base.double();
            }
        }

        result
    }

    #[inline]
    fn to_affine(self) -> Option<(Fp, Fp)> {
        if self.is_zero() {
            None
        } else {
            Some((self.x, self.y))
        }
    }
}

// ---------------------------------------------------------------------------
// MSM public
//
// points  : coordonnées affines (x_le_bytes, y_le_bytes)
// scalars : BigInt<4> little-endian [u64; 4] (depuis ark_ff into_bigint())
//
// Retourne true si le résultat est le point à l'infini
// ---------------------------------------------------------------------------
pub fn sp1_pallas_msm(points: &[([u8; 32], [u8; 32])], scalars: &[[u64; 4]]) -> bool {
    // Vérifie la cohérence avec ark-ec sur les 3 premiers éléments
    #[cfg(not(target_os = "zkvm"))]
    {
        use ark_ec::AffineRepr;
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_ff::BigInt;
        use ark_serialize::CanonicalDeserialize;
        use mina_curves::pasta::{Fp as ArkFp, Pallas, ProjectivePallas};

        if points.len() >= 2 {
            // Trouve 2 points non-nuls
            let non_zero: Vec<_> = points
                .iter()
                .zip(scalars.iter())
                .filter(|((px, py), _)| px != &[0u8; 32] || py != &[0u8; 32])
                .take(2)
                .collect();

            if non_zero.len() >= 2 {
                use ark_serialize::CanonicalSerialize;

                let ((px0, py0), sc0) = non_zero[0];
                let ((px1, py1), sc1) = non_zero[1];

                let p0_x = ArkFp::deserialize_uncompressed(&px0[..]).unwrap();
                let p0_y = ArkFp::deserialize_uncompressed(&py0[..]).unwrap();
                let p1_x = ArkFp::deserialize_uncompressed(&px1[..]).unwrap();
                let p1_y = ArkFp::deserialize_uncompressed(&py1[..]).unwrap();

                let ark_point0 = Pallas::new_unchecked(p0_x, p0_y);
                let ark_point1 = Pallas::new_unchecked(p1_x, p1_y);

                let ark_s0 = BigInt::<4>(*sc0);
                let ark_s1 = BigInt::<4>(*sc1);

                let ark_res =
                    ProjectivePallas::msm_bigint(&[ark_point0, ark_point1], &[ark_s0, ark_s1]);
                let ark_affine = ark_res.into_affine();

                let p0 = PallasPoint::from_affine(Fp::from_le_bytes(px0), Fp::from_le_bytes(py0));
                let p1 = PallasPoint::from_affine(Fp::from_le_bytes(px1), Fp::from_le_bytes(py1));
                let s0_bytes: [u8; 32] = bytemuck::cast(*sc0);
                let s1_bytes: [u8; 32] = bytemuck::cast(*sc1);
                let our_sum = p0.scalar_mul(&s0_bytes).add(p1.scalar_mul(&s1_bytes));

                let mut ark_xb = [0u8; 32];
                ark_affine.x.serialize_compressed(&mut ark_xb[..]).unwrap();

                if let Some((our_x, _)) = our_sum.to_affine() {
                    eprintln!("our x[..4] = {:?}", &our_x.to_le_bytes()[..4]);
                    eprintln!("ark x[..4] = {:?}", &ark_xb[..4]);
                    eprintln!("match: {}", our_x.to_le_bytes() == ark_xb);
                } else {
                    eprintln!("our sum = INFINITY, ark is_zero: {}", ark_affine.is_zero());
                }
            }
        }
    }

    debug_assert_eq!(points.len(), scalars.len());

    // Test point 0 — vérifie que from_affine + add(INFINITY) = lui-même
    if !points.is_empty() {
        let p = PallasPoint::from_affine(
            Fp::from_le_bytes(&points[0].0),
            Fp::from_le_bytes(&points[0].1),
        );
        let p_plus_inf = p.add(PallasPoint::INFINITY);

        // Test doublement : p + p
        let p2 = p.add(p);
        if let Some((x2, y2)) = p2.to_affine() {
            eprintln!("2p.x = {:?}", &x2.to_le_bytes()[..8]);
            eprintln!("2p.y = {:?}", &y2.to_le_bytes()[..8]);
        } else {
            eprintln!("2p = INFINITY (bug!)");
        }
    }

    // Test scalaire 1 : 1*P = P
    if !points.is_empty() {
        let p = PallasPoint::from_affine(
            Fp::from_le_bytes(&points[0].0),
            Fp::from_le_bytes(&points[0].1),
        );
        let p_plus_inf = p.add(PallasPoint::INFINITY);
        eprintln!("p+inf is_zero: {}", p_plus_inf.is_zero());
        match (p.to_affine(), p_plus_inf.to_affine()) {
            (Some((ax, _)), Some((bx, _))) => eprintln!("p.x == (p+inf).x : {}", ax == bx),
            (Some(_), None) => eprintln!("p+inf = INFINITY (bug!)"),
            _ => eprintln!("p = INFINITY (unexpected)"),
        }

        // Test 1*P = P
        let mut one = [0u8; 32];
        one[0] = 1;
        let p1 = p.scalar_mul(&one);
        match (p.to_affine(), p1.to_affine()) {
            (Some((ax, _)), Some((bx, _))) => eprintln!("1*P == P : {}", ax == bx),
            _ => eprintln!("1*P or P is INFINITY (bug!)"),
        }

        // Test 2*P
        let p2 = p.add(p);
        match p2.to_affine() {
            Some((x2, y2)) => eprintln!("2p.x = {:?}", &x2.to_le_bytes()[..4]),
            None => eprintln!("2p = INFINITY (bug!)"),
        }
    }

    let mut acc = PallasPoint::INFINITY;
    for (i, ((px, py), sc)) in points.iter().zip(scalars.iter()).enumerate() {
        let scalar_bytes: [u8; 32] = bytemuck::cast(*sc);
        let p = PallasPoint::from_affine(Fp::from_le_bytes(px), Fp::from_le_bytes(py));
        let contrib = p.scalar_mul(&scalar_bytes);
        acc = acc.add(contrib);
        if i < 3 {
            if let Some((x, _)) = contrib.to_affine() {
                eprintln!("contrib[{}].x = {:?}", i, &x.to_le_bytes()[..8]);
            } else {
                eprintln!("contrib[{}] = INFINITY", i);
            }
        }
    }

    eprintln!("final acc is_zero: {}", acc.is_zero());
    acc.is_zero()
}
