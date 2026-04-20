//! SP1-optimized MSM for Pallas using sys_bigint precompile.
//! Only active when compiled for the SP1 zkVM target.

#![cfg(target_os = "zkvm")]

use crypto_bigint::{Encoding, NonZero, U256, U512};

// ---------------------------------------------------------------------------
// Fp — Optimize for SP1
// ---------------------------------------------------------------------------
const FP_MODULUS: U256 =
    U256::from_be_hex("40000000000000000000000000000000224698fc094cf91b992d30ed00000001");
const FP_MODULUS_LIMBS: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
];
const FP_NONZERO: NonZero<U256> = NonZero::from_uint(FP_MODULUS);

// ---------------------------------------------------------------------------
// Fq — Pallas scalar field (Vesta base field)
// ---------------------------------------------------------------------------
const FQ_MODULUS: U256 =
    U256::from_be_hex("40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001");
const FQ_MODULUS_LIMBS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
];
const FQ_NONZERO: NonZero<U256> = NonZero::from_uint(FQ_MODULUS);

// ---------------------------------------------------------------------------
// Fp field element
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fp(U256);

impl Fp {
    const ZERO: Self = Fp(U256::ZERO);
    const ONE:  Self = Fp(U256::ONE);

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
        let lhs: [u64; 4] = bytemuck::cast(self.0.to_le_bytes());
        let rhs: [u64; 4] = bytemuck::cast(rhs.0.to_le_bytes());
        let mut result = [0u64; 4];
        unsafe {
            sp1_lib::sys_bigint(
                &mut result as *mut [u64; 4],
                0, // OP_MULMOD
                &lhs as *const [u64; 4],
                &rhs as *const [u64; 4],
                &FP_MODULUS_LIMBS as *const [u64; 4],
            );
        }
        Fp(U256::from_le_bytes(bytemuck::cast(result)))
    }

    #[inline(always)]
    fn neg(self) -> Self {
        if self.0 == U256::ZERO { self } else { Fp(FP_MODULUS.wrapping_sub(&self.0)) }
    }

    #[inline(always)]
    fn square(self) -> Self { self.mul(self) }

    fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exp > 0 {
            if exp & 1 == 1 { result = result.mul(base); }
            base = base.square();
            exp >>= 1;
        }
        result
    }

    fn inverse(self) -> Option<Self> {
        if self.0 == U256::ZERO { return None; }
        // Fermat: a^{p-2}
        // p - 2 for Fp
        let exp_bytes = FP_MODULUS.wrapping_sub(&U256::from(2u64));
        let mut result = Self::ONE;
        let mut base = self;
        let bits = 255usize;
        for i in 0..bits {
            let byte = exp_bytes.to_le_bytes()[i / 8];
            if (byte >> (i % 8)) & 1 == 1 {
                result = result.mul(base);
            }
            base = base.square();
        }
        Some(result)
    }

    fn from_le_bytes(b: [u8; 32]) -> Self {
        Fp(U256::from_le_bytes(b))
    }

    fn to_le_bytes(self) -> [u8; 32] {
        self.0.to_le_bytes()
    }
}

// ---------------------------------------------------------------------------
// Pallas point in projective coordinates (X:Y:Z)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct PallasPoint {
    x: Fp,
    y: Fp,
    z: Fp,
}

impl PallasPoint {
    const INFINITY: Self = PallasPoint {
        x: Fp::ONE,
        y: Fp::ONE,
        z: Fp::ZERO,
    };

    fn from_affine(x: Fp, y: Fp) -> Self {
        PallasPoint { x, y, z: Fp::ONE }
    }

    fn is_zero(&self) -> bool {
        self.z.0 == U256::ZERO
    }

    // Complete addition formula for short Weierstrass y²=x³+5 (Pallas b=5)
    fn add(self, rhs: Self) -> Self {
        // Using complete addition from "Complete addition formulas for prime order elliptic curves"
        // (Renes, Costello, Renes 2015)
        let b3 = Fp::from_le_bytes({
            // 3*5 = 15 as field element
            let mut b = [0u8; 32];
            b[0] = 15;
            b
        });

        let (x1, y1, z1) = (self.x, self.y, self.z);
        let (x2, y2, z2) = (rhs.x, rhs.y, rhs.z);

        let t0 = x1.mul(x2);
        let t1 = y1.mul(y2);
        let t2 = z1.mul(z2);
        let t3 = x1.add(y1);
        let t4 = x2.add(y2);
        let t3 = t3.mul(t4);
        let t4 = t0.add(t1);
        let t3 = t3.sub(t4);
        let t4 = x1.add(z1);
        let t5 = x2.add(z2);
        let t4 = t4.mul(t5);
        let t5 = t0.add(t2);
        let t4 = t4.sub(t5);
        let t5 = y1.add(z1);
        let x3 = y2.add(z2);
        let t5 = t5.mul(x3);
        let x3 = t1.add(t2);
        let t5 = t5.sub(x3);
        let z3 = b3.mul(t2);
        let x3 = t4.sub(z3);
        let z3 = x3.add(x3);
        let x3 = x3.add(z3);
        let z3 = t1.sub(x3);
        let x3 = t1.add(x3);
        let y3 = b3.mul(t4);
        let t1 = t2.add(t2);
        let t2 = t1.add(t2);
        let y3 = y3.sub(t2);
        let y3 = y3.sub(t0);
        let t1 = y3.add(y3);
        let y3 = t1.add(y3);
        let t1 = t0.add(t0);
        let t0 = t1.add(t0);
        let t0 = t0.sub(t2);
        let t1 = t4.mul(y3);
        let t2 = t0.mul(y3);
        let y3 = x3.mul(z3);
        let y3 = y3.add(t2);
        let x3 = t3.mul(x3);
        let x3 = x3.sub(t1);
        let z3 = t4.mul(z3);
        let t1 = t3.mul(t0);
        let z3 = z3.add(t1);

        PallasPoint { x: x3, y: y3, z: z3 }
    }

    fn double(self) -> Self {
        self.add(self)
    }

    fn scalar_mul(self, scalar_bytes: &[u64; 4]) -> Self {
        let mut result = Self::INFINITY;
        let mut base = self;
        for limb in scalar_bytes.iter() {
            for bit in 0..64 {
                if (limb >> bit) & 1 == 1 {
                    result = result.add(base);
                }
                base = base.double();
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Public MSM function — replaces G::Group::msm_bigint in ipa.rs
// ---------------------------------------------------------------------------

/// Convert ark Pallas affine point to our Fp representation
fn ark_to_fp(v: &[u8; 32]) -> Fp {
    Fp::from_le_bytes(*v)
}

/// SP1-optimized MSM for Pallas
/// points: affine Pallas points as (x_le, y_le) byte pairs
/// scalars: scalar field elements as 4x u64 limbs (little-endian)
pub fn sp1_pallas_msm(
    points_x: &[[u8; 32]],
    points_y: &[[u8; 32]],
    scalars: &[[u64; 4]],
) -> ([u8; 32], [u8; 32], bool) {
    assert_eq!(points_x.len(), scalars.len());

    let mut acc = PallasPoint::INFINITY;

    for ((px, py), sc) in points_x.iter().zip(points_y.iter()).zip(scalars.iter()) {
        let p = PallasPoint::from_affine(
            Fp::from_le_bytes(*px),
            Fp::from_le_bytes(*py),
        );
        let contribution = p.scalar_mul(sc);
        acc = acc.add(contribution);
    }

    if acc.is_zero() {
        return ([0u8; 32], [0u8; 32], true);
    }

    // Convert back to affine: x = X/Z, y = Y/Z
    let z_inv = acc.z.inverse().unwrap();
    let x_affine = acc.x.mul(z_inv);
    let y_affine = acc.y.mul(z_inv);

    (x_affine.to_le_bytes(), y_affine.to_le_bytes(), false)
}