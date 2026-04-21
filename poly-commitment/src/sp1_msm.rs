//! SP1-optimized MSM for Vesta/Pallas using Pippenger + sys_bigint precompile.
//! Vesta: y² = x³ + 5, BaseField = Fq
//! https://eprint.iacr.org/2015/1060.pdf (addition formula)

use ark_ec::AffineRepr;
use ark_ff::PrimeField;
use crypto_bigint::{Encoding, NonZero, U256, U512};

// ---------------------------------------------------------------------------
// Vesta base field (coordonnées des points IPA)
// ---------------------------------------------------------------------------
const FQ_MODULUS: U256 =
    U256::from_be_hex("40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001");
const FQ_MODULUS_LIMBS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
];

// ---------------------------------------------------------------------------
// Pallas base field
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
// Generic Fp — modular arithmetic with sys_bigint on zkvm
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Fp {
    v: U256,
    m: U256,      // modulus
    ml: [u64; 4], // modulus limbs (little-endian) for sys_bigint
}

impl Fp {
    #[inline]
    fn new(v: U256, m: U256, ml: [u64; 4]) -> Self {
        Fp { v, m, ml }
    }
    #[inline]
    fn zero(m: U256, ml: [u64; 4]) -> Self {
        Fp::new(U256::ZERO, m, ml)
    }
    #[inline]
    fn one(m: U256, ml: [u64; 4]) -> Self {
        Fp::new(U256::ONE, m, ml)
    }
    #[inline]
    fn k(&self, n: u64) -> Self {
        Fp::new(U256::from(n), self.m, self.ml)
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Fp::new(self.v.add_mod(&rhs.v, &self.m), self.m, self.ml)
    }
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Fp::new(self.v.sub_mod(&rhs.v, &self.m), self.m, self.ml)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        if self.v == U256::ZERO {
            self
        } else {
            Fp::new(self.m.wrapping_sub(&self.v), self.m, self.ml)
        }
    }

    #[inline(always)]
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
                    &self.ml as *const [u64; 4],
                );
            }
            return Fp::new(U256::from_le_bytes(bytemuck::cast(result)), self.m, self.ml);
        }
        #[cfg(not(target_os = "zkvm"))]
        {
            let (lo, hi) = self.v.mul_wide(&rhs.v);
            let wide = U512::from((lo, hi));
            let m512 = U512::from((self.m, U256::ZERO));
            let (_, rem) = wide.div_rem(&NonZero::from_uint(m512));
            Fp::new(
                U256::from_le_bytes(rem.to_le_bytes()[..32].try_into().unwrap()),
                self.m,
                self.ml,
            )
        }
    }

    #[inline(always)]
    fn square(self) -> Self {
        self.mul(self)
    }

    fn from_le(b: &[u8; 32], m: U256, ml: [u64; 4]) -> Self {
        Fp::new(U256::from_le_bytes(*b), m, ml)
    }
    fn to_le_bytes(self) -> [u8; 32] {
        self.v.to_le_bytes()
    }
    fn is_zero(&self) -> bool {
        self.v == U256::ZERO
    }
}

// ---------------------------------------------------------------------------
// Elliptic curve point in Jacobian coordinates
// Affine (x,y) → Jacobian (x:y:1)
// Infinity   → (1:1:0)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct Point {
    x: Fp,
    y: Fp,
    z: Fp,
}

impl Point {
    fn infinity(m: U256, ml: [u64; 4]) -> Self {
        Point {
            x: Fp::one(m, ml),
            y: Fp::one(m, ml),
            z: Fp::zero(m, ml),
        }
    }
    fn from_affine(x: Fp, y: Fp) -> Self {
        let one = Fp::one(x.m, x.ml);
        Point { x, y, z: one }
    }
    fn is_zero(&self) -> bool {
        self.z.is_zero()
    }

    /// Jacobian doubling, a=0
    fn double(self) -> Self {
        if self.is_zero() || self.y.is_zero() {
            return Point::infinity(self.x.m, self.x.ml);
        }
        let a = self.x.square();
        let b = self.y.square();
        let c = b.square();
        let x1b = self.x.add(b);
        let d = self.x.k(2).mul(x1b.square().sub(a).sub(c));
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

    /// Full Jacobian addition
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

        if h.is_zero() {
            return if r.is_zero() {
                self.double()
            } else {
                Point::infinity(self.x.m, self.x.ml)
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

    /// Mixed addition: Jacobian self + affine (x2, y2)
    /// Saves 4M vs full Jacobian addition (Z2=1 optimization)
    fn add_affine(self, x2: Fp, y2: Fp) -> Self {
        if self.is_zero() {
            return Point::from_affine(x2, y2);
        }

        let z1z1 = self.z.square();
        let u2 = x2.mul(z1z1);
        let s2 = y2.mul(self.z).mul(z1z1);
        let h = u2.sub(self.x);
        let r = s2.sub(self.y);

        if h.is_zero() {
            return if r.is_zero() {
                self.double()
            } else {
                Point::infinity(self.x.m, self.x.ml)
            };
        }

        let hh = h.square();
        let hhh = h.mul(hh);
        let v = self.x.mul(hh);
        let x3 = r.square().sub(hhh).sub(self.x.k(2).mul(v));
        let y3 = r.mul(v.sub(x3)).sub(self.y.mul(hhh));
        let z3 = h.mul(self.z);
        Point {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    fn neg(self) -> Self {
        if self.is_zero() {
            return self;
        }
        Point {
            x: self.x,
            y: self.y.neg(),
            z: self.z,
        }
    }
}

// ---------------------------------------------------------------------------
// Pippenger MSM
//
// Algorithm:
//   c = window size (bits)
//   For each c-bit window of each scalar:
//     bucket[digit] += point
//   Combine buckets with running sum trick
//   Combine windows: result = sum_w (window_w * 2^(w*c))
// ---------------------------------------------------------------------------
fn pippenger(
    points: &[([u8; 32], [u8; 32])],
    scalars: &[[u64; 4]],
    m: U256,
    ml: [u64; 4],
) -> Point {
    let n = points.len();
    if n == 0 {
        return Point::infinity(m, ml);
    }

    // Window size: same heuristic as ark-ec
    let c = if n < 32 { 3 } else { ln_without_floats(n) + 2 };

    let num_bits = 255usize;
    let num_windows = (num_bits + c - 1) / c;
    let num_buckets = (1usize << c) - 1;

    // Pre-parse all points — skip zero points
    let parsed: Vec<Option<(Fp, Fp)>> = points
        .iter()
        .map(|(px, py)| {
            if px == &[0u8; 32] && py == &[0u8; 32] {
                return None;
            }
            Some((Fp::from_le(px, m, ml), Fp::from_le(py, m, ml)))
        })
        .collect();

    let mut window_sums: Vec<Point> = Vec::with_capacity(num_windows);

    for w in 0..num_windows {
        let mut buckets = vec![Point::infinity(m, ml); num_buckets + 1];

        for (i, sc) in scalars.iter().enumerate() {
            if parsed[i].is_none() {
                continue;
            }

            // Extract c-bit window at position w*c
            let digit = extract_bits(sc, w * c, c);
            if digit == 0 {
                continue;
            }

            let (px, py) = parsed[i].unwrap();
            buckets[digit] = buckets[digit].add_affine(px, py);
        }

        // Sum buckets: sum_{i=1}^{2^c-1} i * bucket[i]
        // = sum_{i=1}^{2^c-1} (sum_{j=i}^{2^c-1} bucket[j])
        // Uses the running-sum trick: 2*(2^c-1) additions
        let mut running_sum = Point::infinity(m, ml);
        let mut window_sum = Point::infinity(m, ml);
        for b in (1..=num_buckets).rev() {
            running_sum = running_sum.add(buckets[b]);
            window_sum = window_sum.add(running_sum);
        }

        window_sums.push(window_sum);
    }

    // Combine windows: result = sum_w window_w * 2^(w*c)
    // Traverse from highest to lowest window
    let lowest = window_sums[0];
    let upper =
        window_sums[1..]
            .iter()
            .rev()
            .fold(Point::infinity(m, ml), |mut total, window_sum| {
                total = total.add(*window_sum);
                for _ in 0..c {
                    total = total.double();
                }
                total
            });
    upper.add(lowest)
}

/// Extract `width` bits from a [u64; 4] little-endian scalar starting at bit `start`
#[inline]
fn extract_bits(scalar: &[u64; 4], start: usize, width: usize) -> usize {
    let limb_idx = start / 64;
    let bit_idx = start % 64;

    if limb_idx >= 4 {
        return 0;
    }

    let lo = scalar[limb_idx] >> bit_idx;

    let hi = if bit_idx > 0 && limb_idx + 1 < 4 {
        scalar[limb_idx + 1] << (64 - bit_idx)
    } else {
        0
    };

    let mask = (1usize << width) - 1;
    ((lo | hi) as usize) & mask
}

/// Integer log2 without floats (same as ark-ec)
fn ln_without_floats(n: usize) -> usize {
    let mut log = 0;
    let mut x = n;
    while x > 1 {
        x >>= 1;
        log += 1;
    }
    log
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
fn sp1_curve_msm(
    points: &[([u8; 32], [u8; 32])],
    scalars: &[[u64; 4]],
    m: U256,
    ml: [u64; 4],
) -> bool {
    debug_assert_eq!(points.len(), scalars.len());

    let result = pippenger(points, scalars, m, ml);
    {
        use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use mina_curves::pasta::{Fq as ArkFq, ProjectiveVesta, Vesta};

        let ark_pts: Vec<Vesta> = points
            .iter()
            .map(|(px, py)| {
                if px == &[0u8; 32] && py == &[0u8; 32] {
                    return Vesta::default();
                }
                Vesta::new_unchecked(
                    ArkFq::deserialize_uncompressed(&px[..]).unwrap(),
                    ArkFq::deserialize_uncompressed(&py[..]).unwrap(),
                )
            })
            .collect();

        let ark_scs: Vec<_> = scalars.iter().map(|s| ark_ff::BigInt::<4>(*s)).collect();
        let ark_res = ProjectiveVesta::msm_bigint(&ark_pts, &ark_scs).into_affine();

        eprintln!(
            "[sp1_msm] n={} ark_is_zero={} our_is_zero={} match={}",
            points.len(),
            ark_res.is_zero(),
            result.is_zero(),
            ark_res.is_zero() == result.is_zero()
        );
    }

    result.is_zero()
}

pub fn sp1_vesta_msm(points: &[([u8; 32], [u8; 32])], scalars: &[[u64; 4]]) -> bool {
    sp1_curve_msm(points, scalars, FQ_MODULUS, FQ_MODULUS_LIMBS)
}

pub fn sp1_pallas_msm(points: &[([u8; 32], [u8; 32])], scalars: &[[u64; 4]]) -> bool {
    sp1_curve_msm(points, scalars, FP_MODULUS, FP_MODULUS_LIMBS)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use ark_ec::{CurveGroup, VariableBaseMSM};
    use ark_ff::UniformRand;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use mina_curves::pasta::{
        Fq as ArkFq, Pallas as ArkPallas, PallasParameters, ProjectivePallas, ProjectiveVesta,
        Vesta as ArkVesta, VestaParameters,
    };

    fn vesta_generator() -> ([u8; 32], [u8; 32]) {
        use ark_ec::short_weierstrass::SWCurveConfig;
        let g = VestaParameters::GENERATOR;
        let mut xb = [0u8; 32];
        let mut yb = [0u8; 32];
        g.x.serialize_uncompressed(&mut xb[..]).unwrap();
        g.y.serialize_uncompressed(&mut yb[..]).unwrap();
        (xb, yb)
    }

    fn ark_vesta_mul(px: &[u8; 32], py: &[u8; 32], sc: [u64; 4]) -> [u8; 32] {
        use mina_curves::pasta::Fq as ArkFq;
        let res = ProjectiveVesta::msm_bigint(
            &[ArkVesta::new_unchecked(
                ArkFq::deserialize_uncompressed(&px[..]).unwrap(),
                ArkFq::deserialize_uncompressed(&py[..]).unwrap(),
            )],
            &[ark_ff::BigInt::<4>(sc)],
        )
        .into_affine();
        let mut xb = [0u8; 32];
        res.x.serialize_uncompressed(&mut xb[..]).unwrap();
        xb
    }

    fn our_vesta_x(p: Point) -> [u8; 32] {
        // Jacobian → affine: x = X/Z²
        let z2 = p.z.square();
        let exp = FQ_MODULUS.wrapping_sub(&U256::from(2u64));
        let mut r = Fp::one(FQ_MODULUS, FQ_MODULUS_LIMBS);
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
    fn test_vesta_1g() {
        let (gx, gy) = vesta_generator();
        let sc = [1u64, 0, 0, 0];
        let res = pippenger(&[(gx, gy)], &[sc], FQ_MODULUS, FQ_MODULUS_LIMBS);
        assert_eq!(our_vesta_x(res), gx, "1*G != G");
    }

    #[test]
    fn test_vesta_2g() {
        let (gx, gy) = vesta_generator();
        let sc = [2u64, 0, 0, 0];
        let res = pippenger(&[(gx, gy)], &[sc], FQ_MODULUS, FQ_MODULUS_LIMBS);
        assert_eq!(
            our_vesta_x(res),
            ark_vesta_mul(&gx, &gy, sc),
            "2*G mismatch"
        );
    }

    #[test]
    fn test_vesta_random() {
        use ark_ec::CurveGroup;
        let mut rng = rand::thread_rng();
        let n = 20;
        let ark_pts: Vec<_> = (0..n)
            .map(|_| ProjectiveVesta::rand(&mut rng).into_affine())
            .collect();
        let ark_scs: Vec<_> = (0..n)
            .map(|_| mina_curves::pasta::Fp::rand(&mut rng).into_bigint())
            .collect();
        let our_pts: Vec<_> = ark_pts
            .iter()
            .map(|p| {
                let mut xb = [0u8; 32];
                let mut yb = [0u8; 32];
                p.x.serialize_uncompressed(&mut xb[..]).unwrap();
                p.y.serialize_uncompressed(&mut yb[..]).unwrap();
                (xb, yb)
            })
            .collect();
        let our_scs: Vec<[u64; 4]> = ark_scs
            .iter()
            .map(|s| s.as_ref().try_into().unwrap())
            .collect();

        let ark_res = ProjectiveVesta::msm_bigint(&ark_pts, &ark_scs).into_affine();
        let our_res = sp1_vesta_msm(&our_pts, &our_scs);
        assert_eq!(our_res, ark_res.is_zero());
    }

    #[test]
    fn test_vesta_full_size() {
        use ark_ec::CurveGroup;
        let mut rng = rand::thread_rng();
        let n = 450;
        let ark_pts: Vec<_> = (0..n)
            .map(|_| ProjectiveVesta::rand(&mut rng).into_affine())
            .collect();
        let ark_scs: Vec<_> = (0..n)
            .map(|_| mina_curves::pasta::Fp::rand(&mut rng).into_bigint())
            .collect();
        let our_pts: Vec<_> = ark_pts
            .iter()
            .map(|p| {
                let mut xb = [0u8; 32];
                let mut yb = [0u8; 32];
                p.x.serialize_uncompressed(&mut xb[..]).unwrap();
                p.y.serialize_uncompressed(&mut yb[..]).unwrap();
                (xb, yb)
            })
            .collect();
        let our_scs: Vec<[u64; 4]> = ark_scs
            .iter()
            .map(|s: &ark_ff::BigInt<4>| s.as_ref().try_into().unwrap())
            .collect();

        let ark_res = ProjectiveVesta::msm_bigint(&ark_pts, &ark_scs).into_affine();
        let our_res = sp1_vesta_msm(&our_pts, &our_scs);
        assert_eq!(our_res, ark_res.is_zero());
    }

    #[test]
    fn test_fp_mul_basic() {
        // 2 * 3 = 6
        let two = Fp::new(U256::from(2u64), FQ_MODULUS, FQ_MODULUS_LIMBS);
        let three = Fp::new(U256::from(3u64), FQ_MODULUS, FQ_MODULUS_LIMBS);
        let six = two.mul(three);
        assert_eq!(six.v, U256::from(6u64), "2*3 != 6, got {:?}", six.v);

        // (p-1) * 1 = p-1
        let pm1 = Fp::new(
            FQ_MODULUS.wrapping_sub(&U256::ONE),
            FQ_MODULUS,
            FQ_MODULUS_LIMBS,
        );
        let one = Fp::new(U256::ONE, FQ_MODULUS, FQ_MODULUS_LIMBS);
        let res = pm1.mul(one);
        assert_eq!(res.v, FQ_MODULUS.wrapping_sub(&U256::ONE), "(p-1)*1 failed");
    }

    #[test]
    fn test_vesta_with_zero_scalars() {
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_ff::UniformRand;
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use mina_curves::pasta::{Fp as ArkFp, ProjectiveVesta, Vesta};

        let mut rng = rand::thread_rng();
        let ark_pts: Vec<_> = (0..10)
            .map(|_| ProjectiveVesta::rand(&mut rng).into_affine())
            .collect();

        // Mix de scalaires normaux et zéros
        let scalars: Vec<[u64; 4]> = vec![
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [3, 0, 0, 0],
            [0, 0, 0, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [5, 0, 0, 0],
            [0, 0, 0, 0],
        ];

        let our_pts: Vec<_> = ark_pts
            .iter()
            .map(|p| {
                let mut xb = [0u8; 32];
                let mut yb = [0u8; 32];
                p.x.serialize_uncompressed(&mut xb[..]).unwrap();
                p.y.serialize_uncompressed(&mut yb[..]).unwrap();
                (xb, yb)
            })
            .collect();

        let ark_bigints: Vec<_> = scalars.iter().map(|s| ark_ff::BigInt::<4>(*s)).collect();
        let ark_res = ProjectiveVesta::msm_bigint(&ark_pts, &ark_bigints).into_affine();
        let our_res = sp1_vesta_msm(&our_pts, &scalars);
        assert_eq!(our_res, ark_res.is_zero());
    }

    #[test]
    fn test_extract_bits() {
        // Test tous les bits à 1 sur 256 bits
        let sc = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];

        for c in [10usize, 17] {
            for start in (0..256).step_by(c) {
                let d = extract_bits(&sc, start, c);
                // Bits disponibles à partir de `start`
                let available = 256usize.saturating_sub(start);
                let expected = (1usize << available.min(c)) - 1;
                assert_eq!(
                    d, expected,
                    "extract_bits failed at start={} c={}: got {} expected {}",
                    start, c, d, expected
                );
            }
        }

        // scalar=1, seul le bit 0 est à 1
        let sc_one = [1u64, 0, 0, 0];
        assert_eq!(extract_bits(&sc_one, 0, 10), 1);
        assert_eq!(extract_bits(&sc_one, 10, 10), 0);
        assert_eq!(extract_bits(&sc_one, 20, 10), 0);
    }

    #[test]
    fn test_vesta_large() {
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_ff::UniformRand;
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use mina_curves::pasta::{Fp as ArkFp, ProjectiveVesta, Vesta};

        let mut rng = rand::thread_rng();
        let n = 32850;

        let ark_pts: Vec<_> = (0..n)
            .map(|_| ProjectiveVesta::rand(&mut rng).into_affine())
            .collect();
        let ark_scs: Vec<_> = (0..n)
            .map(|_| mina_curves::pasta::Fp::rand(&mut rng).into_bigint())
            .collect();
        let our_pts: Vec<_> = ark_pts
            .iter()
            .map(|p| {
                let mut xb = [0u8; 32];
                let mut yb = [0u8; 32];
                p.x.serialize_uncompressed(&mut xb[..]).unwrap();
                p.y.serialize_uncompressed(&mut yb[..]).unwrap();
                (xb, yb)
            })
            .collect();
        let our_scs: Vec<[u64; 4]> = ark_scs
            .iter()
            .map(|s| s.as_ref().try_into().unwrap())
            .collect();

        let ark_res = ProjectiveVesta::msm_bigint(&ark_pts, &ark_scs).into_affine();
        let our_res = sp1_vesta_msm(&our_pts, &our_scs);
        assert_eq!(our_res, ark_res.is_zero());
    }

    #[test]
    fn test_vesta_ipa_fixture() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
        let data = std::fs::read(dir.join("msm_fixture.bin")).unwrap();
        let (points, scalars): (Vec<([u8; 32], [u8; 32])>, Vec<[u64; 4]>) =
            bincode::deserialize(&data).unwrap();

        // ark reference
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_serialize::CanonicalDeserialize;
        use mina_curves::pasta::{Fq as ArkFq, ProjectiveVesta, Vesta};

        let ark_pts: Vec<Vesta> = points
            .iter()
            .map(|(px, py)| {
                if px == &[0u8; 32] {
                    return Vesta::default();
                }
                Vesta::new_unchecked(
                    ArkFq::deserialize_uncompressed(&px[..]).unwrap(),
                    ArkFq::deserialize_uncompressed(&py[..]).unwrap(),
                )
            })
            .collect();
        let ark_scs: Vec<_> = scalars.iter().map(|s| ark_ff::BigInt::<4>(*s)).collect();
        let ark_res = ProjectiveVesta::msm_bigint(&ark_pts, &ark_scs).into_affine();

        let our_res = sp1_vesta_msm(&points, &scalars);

        eprintln!("ark_is_zero={} our_is_zero={}", ark_res.is_zero(), our_res);
        assert_eq!(our_res, ark_res.is_zero());
    }
}
