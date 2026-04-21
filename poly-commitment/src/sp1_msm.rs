//! SP1-optimized MSM for Pallas using crypto-bigint + sys_bigint precompile.
//! Pallas: y² = x³ + 5, BaseField = Fp
//! Coordinate system: Jacobian (X:Y:Z) where (x,y) = (X/Z², Y/Z³)

use crypto_bigint::{Encoding, NonZero, U256, U512};

const FP_MODULUS: U256 =
    U256::from_be_hex("40000000000000000000000000000000224698fc094cf91b992d30ed00000001");
const FP_MODULUS_LIMBS: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
];

// Vesta base field = Pallas scalar field
// Fq modulus = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
const FQ_MODULUS: U256 =
    U256::from_be_hex("40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001");
const FQ_MODULUS_LIMBS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fp(U256);

impl Fp {
    const ZERO: Self = Fp(U256::ZERO);
    const ONE: Self = Fp(U256::ONE);

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Fp(self.0.add_mod(&rhs.0, &FP_MODULUS))
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Fp(self.0.sub_mod(&rhs.0, &FP_MODULUS))
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(target_os = "zkvm")]
        {
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
    fn square(self) -> Self {
        self.mul(self)
    }

    fn from_le_bytes(b: &[u8; 32]) -> Self {
        Fp(U256::from_le_bytes(*b))
    }
    fn to_le_bytes(self) -> [u8; 32] {
        self.0.to_le_bytes()
    }

    fn k(n: u64) -> Self {
        Fp(U256::from(n))
    }
}

// ---------------------------------------------------------------------------
// Pallas in Jacobian coordinates: (X:Y:Z) → affine (X/Z², Y/Z³)
// Infinity = (1:1:0)
// Curve: y² = x³ + 5  (a=0, b=5)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct Pallas {
    x: Fp,
    y: Fp,
    z: Fp,
}

impl Pallas {
    const INFINITY: Self = Pallas {
        x: Fp::ONE,
        y: Fp::ONE,
        z: Fp::ZERO,
    };

    fn from_affine(x: Fp, y: Fp) -> Self {
        Pallas { x, y, z: Fp::ONE }
    }

    fn is_zero(&self) -> bool {
        self.z == Fp::ZERO
    }

    /// Jacobian point doubling, a=0
    /// Cost: 1I + 3S + 6M (no inversions needed in projective)
    fn double(self) -> Self {
        if self.is_zero() || self.y == Fp::ZERO {
            return Self::INFINITY;
        }

        // a = X1²
        let a = self.x.square();
        // b = Y1²
        let b = self.y.square();
        // c = b²
        let c = b.square();
        // d = 2*((X1+b)² - a - c)
        let d = Fp::k(2).mul(self.x.add(b).square().sub(a).sub(c));
        // e = 3*a  (a_coeff=0, so skip a*Z1⁴)
        let e = Fp::k(3).mul(a);
        // f = e²
        let f = e.square();
        // X3 = f - 2*d
        let x3 = f.sub(Fp::k(2).mul(d));
        // Y3 = e*(d - X3) - 8*c
        let y3 = e.mul(d.sub(x3)).sub(Fp::k(8).mul(c));
        // Z3 = 2*Y1*Z1
        let z3 = Fp::k(2).mul(self.y).mul(self.z);

        Pallas {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// Full Jacobian-Jacobian addition
    /// Cost: 12M + 4S
    fn add(self, rhs: Self) -> Self {
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }

        // Z1² , Z2²
        let z1z1 = self.z.square();
        let z2z2 = rhs.z.square();
        // U1 = X1*Z2², U2 = X2*Z1²
        let u1 = self.x.mul(z2z2);
        let u2 = rhs.x.mul(z1z1);
        // S1 = Y1*Z2*Z2², S2 = Y2*Z1*Z1²
        let s1 = self.y.mul(rhs.z).mul(z2z2);
        let s2 = rhs.y.mul(self.z).mul(z1z1);
        // H = U2 - U1,  R = S2 - S1
        let h = u2.sub(u1);
        let r = s2.sub(s1);

        if h == Fp::ZERO {
            return if r == Fp::ZERO {
                self.double()
            } else {
                Self::INFINITY
            };
        }

        let hh = h.square(); // H²
        let hhh = h.mul(hh); // H³
        let v = u1.mul(hh); // V = U1*H²

        // X3 = R² - H³ - 2*V
        let x3 = r.square().sub(hhh).sub(Fp::k(2).mul(v));
        // Y3 = R*(V - X3) - S1*H³
        let y3 = r.mul(v.sub(x3)).sub(s1.mul(hhh));
        // Z3 = H*Z1*Z2
        let z3 = h.mul(self.z).mul(rhs.z);

        Pallas {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// Mixed addition: Jacobian self + affine (x2, y2)
    /// Cost: 8M + 3S (cheaper than full Jacobian-Jacobian)
    fn add_affine(self, x2: Fp, y2: Fp) -> Self {
        if self.is_zero() {
            return Self::from_affine(x2, y2);
        }

        let z1z1 = self.z.square(); // Z1²
        let u2 = x2.mul(z1z1); // U2 = X2*Z1²
        let s2 = y2.mul(self.z).mul(z1z1); // S2 = Y2*Z1*Z1²
        let h = u2.sub(self.x); // H = U2 - X1
        let r = s2.sub(self.y); // R = S2 - Y1

        if h == Fp::ZERO {
            return if r == Fp::ZERO {
                self.double()
            } else {
                Self::INFINITY
            };
        }

        let hh = h.square();
        let hhh = h.mul(hh);
        let v = self.x.mul(hh);

        let x3 = r.square().sub(hhh).sub(Fp::k(2).mul(v));
        let y3 = r.mul(v.sub(x3)).sub(self.y.mul(hhh));
        let z3 = h.mul(self.z);

        Pallas {
            x: x3,
            y: y3,
            z: z3,
        }
    }

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
}

// ---------------------------------------------------------------------------
// Public MSM
// ---------------------------------------------------------------------------
pub fn sp1_vesta_msm(points: &[([u8; 32], [u8; 32])], scalars: &[[u64; 4]]) -> bool {
    sp1_curve_msm(points, scalars, &FQ_MODULUS, &FQ_MODULUS_LIMBS)
}

pub fn sp1_pallas_msm(points: &[([u8; 32], [u8; 32])], scalars: &[[u64; 4]]) -> bool {
    sp1_curve_msm(points, scalars, &FP_MODULUS, &FP_MODULUS_LIMBS)
}

fn sp1_curve_msm(
    points: &[([u8; 32], [u8; 32])],
    scalars: &[[u64; 4]],
    modulus: &U256,
    modulus_limbs: &[u64; 4],
) -> bool {
    debug_assert_eq!(points.len(), scalars.len());

    // ------------------------------------------------------------------
    // Inner Fp type parametrized by modulus
    // ------------------------------------------------------------------
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct Fp {
        v: U256,
        modulus: U256,
        modulus_limbs: [u64; 4],
    }

    impl Fp {
        fn zero(modulus: &U256, modulus_limbs: &[u64; 4]) -> Self {
            Fp {
                v: U256::ZERO,
                modulus: *modulus,
                modulus_limbs: *modulus_limbs,
            }
        }
        fn one(modulus: &U256, modulus_limbs: &[u64; 4]) -> Self {
            Fp {
                v: U256::ONE,
                modulus: *modulus,
                modulus_limbs: *modulus_limbs,
            }
        }
        fn from_le(b: &[u8; 32], modulus: &U256, modulus_limbs: &[u64; 4]) -> Self {
            Fp {
                v: U256::from_le_bytes(*b),
                modulus: *modulus,
                modulus_limbs: *modulus_limbs,
            }
        }
        fn k(&self, n: u64) -> Self {
            Fp {
                v: U256::from(n),
                modulus: self.modulus,
                modulus_limbs: self.modulus_limbs,
            }
        }
        fn add(self, rhs: Self) -> Self {
            Fp {
                v: self.v.add_mod(&rhs.v, &self.modulus),
                ..self
            }
        }
        fn sub(self, rhs: Self) -> Self {
            Fp {
                v: self.v.sub_mod(&rhs.v, &self.modulus),
                ..self
            }
        }
        fn mul(self, rhs: Self) -> Self {
            #[cfg(target_os = "zkvm")]
            {
                let lhs: [u64; 4] = bytemuck::cast(self.v.to_le_bytes());
                let rhs_l: [u64; 4] = bytemuck::cast(rhs.v.to_le_bytes());
                let mut result = [0u64; 4];
                #[allow(unsafe_code)]
                unsafe {
                    sp1_lib::sys_bigint(
                        &mut result as *mut [u64; 4],
                        0,
                        &lhs as *const [u64; 4],
                        &rhs_l as *const [u64; 4],
                        &self.modulus_limbs as *const [u64; 4],
                    );
                }
                return Fp {
                    v: U256::from_le_bytes(bytemuck::cast(result)),
                    ..self
                };
            }
            #[cfg(not(target_os = "zkvm"))]
            {
                let (lo, hi) = self.v.mul_wide(&rhs.v);
                let wide = U512::from((lo, hi));
                let m512 = U512::from((self.modulus, U256::ZERO));
                let (_, rem) = wide.div_rem(&NonZero::from_uint(m512));
                Fp {
                    v: U256::from_le_bytes(rem.to_le_bytes()[..32].try_into().unwrap()),
                    ..self
                }
            }
        }
        fn square(self) -> Self {
            self.mul(self)
        }
        fn to_le_bytes(self) -> [u8; 32] {
            self.v.to_le_bytes()
        }
    }

    // ------------------------------------------------------------------
    // Point in Jacobian coordinates
    // ------------------------------------------------------------------
    #[derive(Clone, Copy)]
    struct Point {
        x: Fp,
        y: Fp,
        z: Fp,
    }

    impl Point {
        fn infinity(modulus: &U256, modulus_limbs: &[u64; 4]) -> Self {
            Point {
                x: Fp::one(modulus, modulus_limbs),
                y: Fp::one(modulus, modulus_limbs),
                z: Fp::zero(modulus, modulus_limbs),
            }
        }
        fn from_affine(x: Fp, y: Fp) -> Self {
            let one = Fp::one(&x.modulus, &x.modulus_limbs);
            Point { x, y, z: one }
        }
        fn is_zero(&self) -> bool {
            self.z.v == U256::ZERO
        }

        fn double(self) -> Self {
            if self.is_zero() || self.y.v == U256::ZERO {
                return Point::infinity(&self.x.modulus, &self.x.modulus_limbs);
            }
            let a = self.x.square();
            let b = self.y.square();
            let c = b.square();
            let d = self.x.k(2).mul(self.x.add(b).square().sub(a).sub(c));
            let e = self.x.k(3).mul(a);
            let f = e.square();
            let x3 = f.sub(self.x.k(2).mul(d));
            let y3 = e.mul(d.sub(x3)).sub(self.x.k(8).mul(c));
            let z3 = self.x.k(2).mul(self.y).mul(self.z);
            Point {
                x: x3,
                y: y3,
                z: z3,
            }
        }

        fn add(self, rhs: Self) -> Self {
            if self.is_zero() {
                return rhs;
            }
            if rhs.is_zero() {
                return self;
            }

            let z1z1 = self.z.square();
            let z2z2 = rhs.z.square();
            let u1 = self.x.mul(z2z2);
            let u2 = rhs.x.mul(z1z1);
            let s1 = self.y.mul(rhs.z).mul(z2z2);
            let s2 = rhs.y.mul(self.z).mul(z1z1);
            let h = u2.sub(u1);
            let r = s2.sub(s1);

            if h.v == U256::ZERO {
                return if r.v == U256::ZERO {
                    self.double()
                } else {
                    Point::infinity(&self.x.modulus, &self.x.modulus_limbs)
                };
            }

            let hh = h.square();
            let hhh = h.mul(hh);
            let v = u1.mul(hh);
            let x3 = r.square().sub(hhh).sub(self.x.k(2).mul(v));
            let y3 = r.mul(v.sub(x3)).sub(s1.mul(hhh));
            let z3 = h.mul(self.z).mul(rhs.z);

            Point {
                x: x3,
                y: y3,
                z: z3,
            }
        }

        fn scalar_mul(self, scalar_bytes: &[u8; 32]) -> Self {
            let mut result = Point::infinity(&self.x.modulus, &self.x.modulus_limbs);
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
    }

    // ------------------------------------------------------------------
    // MSM
    // ------------------------------------------------------------------
    let mut acc = Point::infinity(modulus, modulus_limbs);

    for ((px, py), sc) in points.iter().zip(scalars.iter()) {
        if px == &[0u8; 32] && py == &[0u8; 32] {
            continue;
        }
        let scalar_bytes: [u8; 32] = bytemuck::cast(*sc);
        let p = Point::from_affine(
            Fp::from_le(px, modulus, modulus_limbs),
            Fp::from_le(py, modulus, modulus_limbs),
        );
        acc = acc.add(p.scalar_mul(&scalar_bytes));
    }

    acc.is_zero()
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{CurveGroup, VariableBaseMSM};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use mina_curves::pasta::{
        Fp as ArkFp, Pallas as ArkPallas, PallasParameters, ProjectivePallas,
    };

    fn generator() -> ([u8; 32], [u8; 32]) {
        use ark_ec::short_weierstrass::SWCurveConfig;
        let g = PallasParameters::GENERATOR;
        let mut xb = [0u8; 32];
        let mut yb = [0u8; 32];
        g.x.serialize_uncompressed(&mut xb[..]).unwrap();
        g.y.serialize_uncompressed(&mut yb[..]).unwrap();
        (xb, yb)
    }

    fn ark_mul(px: &[u8; 32], py: &[u8; 32], sc: [u64; 4]) -> [u8; 32] {
        let res = ProjectivePallas::msm_bigint(
            &[ArkPallas::new_unchecked(
                ArkFp::deserialize_uncompressed(&px[..]).unwrap(),
                ArkFp::deserialize_uncompressed(&py[..]).unwrap(),
            )],
            &[ark_ff::BigInt::<4>(sc)],
        )
        .into_affine();
        let mut xb = [0u8; 32];
        res.x.serialize_uncompressed(&mut xb[..]).unwrap();
        xb
    }

    fn our_x(p: Pallas) -> [u8; 32] {
        // Convert Jacobian to affine x = X/Z²
        let z2 = p.z.square();
        let exp = FP_MODULUS.wrapping_sub(&U256::from(2u64));
        let mut r = Fp::ONE;
        let mut b = z2;
        for i in 0..256 {
            let byte = exp.to_le_bytes()[i / 8];
            if (byte >> (i % 8)) & 1 == 1 {
                r = r.mul(b);
            }
            b = b.square();
        }
        p.x.mul(r).to_le_bytes()
    }

    #[test]
    fn test_msm_5g_7g() {
        let (gx, gy) = generator();
        assert!(!sp1_pallas_msm(
            &[(gx, gy), (gx, gy)],
            &[[5, 0, 0, 0], [7, 0, 0, 0]]
        ));
    }

    #[test]
    fn test_msm_many_random() {
        use ark_ec::AffineRepr;
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_ff::UniformRand;
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use mina_curves::pasta::{Fp as ArkFp, Pallas as ArkPallas, ProjectivePallas};

        let mut rng = rand::thread_rng();
        let n = 10;

        let ark_points: Vec<_> = (0..n)
            .map(|_| ProjectivePallas::rand(&mut rng).into_affine())
            .collect();

        let scalars: Vec<[u64; 4]> = (0..n)
            .map(|_| [rand::random::<u64>() % 1000, 0, 0, 0])
            .collect();

        // Nos points en bytes
        let our_points: Vec<([u8; 32], [u8; 32])> = ark_points
            .iter()
            .map(|p| {
                let mut xb = [0u8; 32];
                let mut yb = [0u8; 32];
                p.x.serialize_uncompressed(&mut xb[..]).unwrap();
                p.y.serialize_uncompressed(&mut yb[..]).unwrap();
                (xb, yb)
            })
            .collect();

        // ark MSM
        let ark_bases: Vec<ArkPallas> = ark_points.clone();
        let ark_bigints: Vec<_> = scalars.iter().map(|s| ark_ff::BigInt::<4>(*s)).collect();
        let ark_res = ProjectivePallas::msm_bigint(&ark_bases, &ark_bigints).into_affine();

        // notre MSM
        let our_res = sp1_pallas_msm(&our_points, &scalars);

        let mut ark_xb = [0u8; 32];
        if !ark_res.is_zero() {
            ark_res.x.serialize_uncompressed(&mut ark_xb[..]).unwrap();
        }

        eprintln!("ark is_zero: {}", ark_res.is_zero());
        eprintln!("our is_zero: {}", our_res);

        assert_eq!(our_res, ark_res.is_zero(), "MSM result mismatch");
    }

    #[test]
    fn test_msm_with_large_scalars() {
        use ark_ec::AffineRepr;
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_ff::PrimeField;
        use ark_ff::UniformRand;
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use mina_curves::pasta::{Fp as ArkFp, Fq, Pallas as ArkPallas, ProjectivePallas};

        let mut rng = rand::thread_rng();
        let n = 10;

        let ark_points: Vec<_> = (0..n)
            .map(|_| ProjectivePallas::rand(&mut rng).into_affine())
            .collect();

        // Scalaires vraiment aléatoires (pleine taille 256 bits)
        let ark_scalars: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
        let ark_bigints: Vec<_> = ark_scalars.iter().map(|s| s.into_bigint()).collect();
        let scalars: Vec<[u64; 4]> = ark_bigints
            .iter()
            .map(|b: &ark_ff::BigInt<4>| b.as_ref().try_into().unwrap())
            .collect();

        let our_points: Vec<([u8; 32], [u8; 32])> = ark_points
            .iter()
            .map(|p| {
                let mut xb = [0u8; 32];
                let mut yb = [0u8; 32];
                p.x.serialize_uncompressed(&mut xb[..]).unwrap();
                p.y.serialize_uncompressed(&mut yb[..]).unwrap();
                (xb, yb)
            })
            .collect();

        let ark_res = ProjectivePallas::msm_bigint(&ark_points, &ark_bigints).into_affine();
        let our_res = sp1_pallas_msm(&our_points, &scalars);

        assert_eq!(our_res, ark_res.is_zero());
    }

    #[test]
    fn test_negative_scalar() {
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_ff::One;
        use ark_ff::PrimeField;
        use ark_ff::UniformRand;
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use mina_curves::pasta::{Fp as ArkFp, Fq, Pallas as ArkPallas, ProjectivePallas};

        let mut rng = rand::thread_rng();
        let ark_p = ProjectivePallas::rand(&mut rng).into_affine();
        let mut px = [0u8; 32];
        let mut py = [0u8; 32];
        ark_p.x.serialize_uncompressed(&mut px[..]).unwrap();
        ark_p.y.serialize_uncompressed(&mut py[..]).unwrap();

        // Scalaire "négatif" : p-1 en représentation BigInt
        let neg_one = (-Fq::one()).into_bigint();
        let sc: [u64; 4] = neg_one.as_ref().try_into().unwrap();

        let ark_res = ProjectivePallas::msm_bigint(&[ark_p], &[neg_one]).into_affine();
        let our = Pallas::from_affine(Fp::from_le_bytes(&px), Fp::from_le_bytes(&py))
            .scalar_mul(&bytemuck::cast(sc));

        let mut ark_xb = [0u8; 32];
        ark_res.x.serialize_uncompressed(&mut ark_xb[..]).unwrap();

        eprintln!("our x[..4] = {:?}", &our_x(our)[..4]);
        eprintln!("ark x[..4] = {:?}", &ark_xb[..4]);
        assert_eq!(our_x(our), ark_xb, "(p-1)*G mismatch");
    }

    #[test]
    fn test_msm_full_size_random_scalars() {
        use ark_ec::AffineRepr;
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_ff::One;
        use ark_ff::PrimeField;
        use ark_ff::UniformRand;
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use mina_curves::pasta::{Fp as ArkFp, Fq, Pallas as ArkPallas, ProjectivePallas};

        let mut rng = rand::thread_rng();
        let n = 428; // même taille que dans le test

        let ark_points: Vec<_> = (0..n)
            .map(|_| ProjectivePallas::rand(&mut rng).into_affine())
            .collect();

        // Scalaires pleine taille incluant négatifs
        let ark_scalars: Vec<_> = (0..n).map(|_| Fq::rand(&mut rng).into_bigint()).collect();
        let scalars: Vec<[u64; 4]> = ark_scalars
            .iter()
            .map(|b| b.as_ref().try_into().unwrap())
            .collect();

        // Inclut aussi des zéros
        let mut our_points: Vec<([u8; 32], [u8; 32])> = ark_points
            .iter()
            .map(|p| {
                let mut xb = [0u8; 32];
                let mut yb = [0u8; 32];
                p.x.serialize_uncompressed(&mut xb[..]).unwrap();
                p.y.serialize_uncompressed(&mut yb[..]).unwrap();
                (xb, yb)
            })
            .collect();

        // Ajoute quelques points à l'infini (padding comme dans ipa.rs)
        for _ in 0..20 {
            our_points.push(([0u8; 32], [0u8; 32]));
        }
        let mut pad_scalars = scalars.clone();
        for _ in 0..20 {
            pad_scalars.push([0u64; 4]);
        }

        let ark_res = ProjectivePallas::msm_bigint(&ark_points, &ark_scalars).into_affine();
        let our_res = sp1_pallas_msm(&our_points, &pad_scalars);

        eprintln!("ark is_zero: {}", ark_res.is_zero());
        eprintln!("our is_zero: {}", our_res);
        assert_eq!(our_res, ark_res.is_zero());
    }
}
