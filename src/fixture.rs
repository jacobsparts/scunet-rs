//! The golden fixtures: input and expected output, as `tools/reference.py` produced them.
//!
//! The container is deliberately trivial - a magic, a small header of u32s, then the
//! input plane and the expected output plane as f32 - so a fixture can be inspected
//! with `xxd` and a truncated or half-written file produces a named error rather than
//! a wrong number. `tools/make_fixture.py` packs the `.npy` pair the reference writes
//! into this form.
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"SCUF";

pub struct Fixture {
    pub version: u32,
    pub variant: String,
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub win: usize,
    pub input: Vec<f32>,
    pub expected: Vec<f32>,
}

fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn f32s(b: &[u8], off: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            f32::from_le_bytes([b[off + 4 * i], b[off + 4 * i + 1], b[off + 4 * i + 2], b[off + 4 * i + 3]])
        })
        .collect()
}

impl Fixture {
    pub fn load(path: impl AsRef<Path>) -> Result<Fixture, String> {
        let path = path.as_ref();
        let b = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        if b.len() < 40 || &b[..4] != MAGIC {
            return Err(format!("{}: not a scunet fixture (want 4-byte magic SCUF)", path.display()));
        }
        let version = u32le(&b, 4);
        if version != 1 {
            return Err(format!("{}: fixture version {version}, this engine reads 1", path.display()));
        }
        let h = u32le(&b, 8) as usize;
        let w = u32le(&b, 12) as usize;
        let c = u32le(&b, 16) as usize;
        let win = u32le(&b, 20) as usize;
        let vlen = u32le(&b, 24) as usize;
        if b.len() < 28 + vlen {
            return Err(format!("{}: the variant string claims {vlen} bytes, the file is shorter", path.display()));
        }
        let variant = String::from_utf8(b[28..28 + vlen].to_vec())
            .map_err(|e| format!("{}: variant is not utf-8: {e}", path.display()))?;
        let off = 28 + vlen;
        let need = off + 4 * (h * w * c * 2);
        if b.len() != need {
            return Err(format!(
                "{}: {need} bytes expected for {h}x{w}x{c} in and out, {} present",
                path.display(),
                b.len()
            ));
        }
        Ok(Fixture {
            version,
            variant,
            h,
            w,
            c,
            win,
            input: f32s(&b, off, h * w * c),
            expected: f32s(&b, off + 4 * h * w * c, h * w * c),
        })
    }

    /// The worst absolute difference, where it is, and the mean - the three numbers
    /// a parity report needs, since "0.31" and "0.31 at one pixel in the top-left
    /// corner" call for different responses.
    pub fn compare(&self, got: &[f32]) -> (f32, usize, f32) {
        assert_eq!(got.len(), self.expected.len(), "backend returned the wrong number of pixels");
        let mut worst = 0.0f32;
        let mut at = 0usize;
        let mut mean = 0.0f64;
        for (i, (a, b)) in self.expected.iter().zip(got.iter()).enumerate() {
            let d = (a - b).abs();
            mean += d as f64;
            if d > worst {
                worst = d;
                at = i;
            }
        }
        (worst, at, (mean / got.len() as f64) as f32)
    }

    /// `index` as (y, x, channel), for the message that accompanies a failure.
    pub fn locate(&self, idx: usize) -> (usize, usize, usize) {
        let plane = self.h * self.w;
        let c = idx / plane;
        let rem = idx % plane;
        (rem / self.w, rem % self.w, c)
    }
}
