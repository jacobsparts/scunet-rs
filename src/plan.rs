//! The geometry of one forward pass, computed once and shared by both backends.
//!
//! SCUNet's geometry is small enough to state in full, which is why this engine can
//! run any image in ONE pass while its Swin2SR sibling cannot:
//!
//! * The input is replicated (edge-clamped, as `nn.ReplicationPad2d`) to the next
//!   multiple of 64 in each axis. 64 is the window size (8) times the three stride-2
//!   downsamples, so after padding every stage's plane is an exact multiple of the
//!   window and no window ever straddles the image border.
//! * The output is cropped back to the original `h`x`w`.
//!
//! WHY THAT IS EXACTLY EQUIVALENT TO PADDING WITH REPLICATED BORDERS, AND WHY IT IS
//! NOT TILING. A convolution over a replicated edge reads the same values a
//! convolution over the padded tensor's interior does, and window attention inside a
//! padded window sees the replicated edge too - so running one padded forward gives
//! the same answer as the reference's `utils_model.test_mode` would for the region
//! that covers the image. Swin2SR cannot do this because its padding is a REFLECTION
//! of the whole plane, which a per-tile derivation would have to redo; replication
//! has no such coupling, so no tiling and no overlap bookkeeping is needed here at
//! all. What the engine does NOT do is window a >512px input the way the reference's
//! test_mode does; it runs the whole plane, which the reference's own code path also
//! does when `test_mode` picks a single window.

/// The padded geometry of one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The input image, before padding.
    pub h: usize,
    pub w: usize,
    /// The plane every stage runs on: `ceil(h/64)*64` by `ceil(w/64)*64`.
    pub hp: usize,
    pub wp: usize,
    /// The window edge, 8 for every published checkpoint.
    pub win: usize,
}

/// The multiple of the window size the input is padded to. 64 is not a magic number:
/// it is `3` stride-2 stages times the window, and it is what the reference's
/// `paddingBottom = int(np.ceil(h / 64) * 64 - h)` computes.
pub const PAD_TO: usize = 64;

impl Plan {
    pub fn new(h: usize, w: usize, win: usize) -> Plan {
        let hp = h.div_ceil(PAD_TO) * PAD_TO;
        let wp = w.div_ceil(PAD_TO) * PAD_TO;
        Plan { h, w, hp, wp, win }
    }

    /// Pixels in one padded plane.
    pub fn plane(&self) -> usize {
        self.hp * self.wp
    }

    /// Windows along each axis at a stage that has been downsampled `dn` times.
    pub fn windows(&self, dn: usize) -> (usize, usize) {
        ((self.hp >> dn) / self.win, (self.wp >> dn) / self.win)
    }

}

/// Replicate-pad `[c][h][w]` to `[c][hp][wp]`, as `nn.ReplicationPad2d((0, right, 0, bottom))`.
///
/// Edge clamping rather than mirroring: the added columns repeat the image's LAST
/// column and the added rows its LAST row. `tools/reference.py` uses `np.pad(...,
/// mode="edge")` and this must agree with it exactly, which is checked in
/// `tests/parity.rs` on an 80x64 fixture where both axes pad.
pub fn pad_replicate(x: &[f32], c: usize, h: usize, w: usize, hp: usize, wp: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * hp * wp];
    for ch in 0..c {
        for y in 0..hp {
            let sy = if y < h { y } else { h - 1 };
            let src = &x[(ch * h + sy) * w..(ch * h + sy) * w + w];
            let dst = &mut out[(ch * hp + y) * wp..(ch * hp + y) * wp + wp];
            dst[..w].copy_from_slice(src);
            let last = src[w - 1];
            for v in dst[w..].iter_mut() {
                *v = last;
            }
        }
    }
    out
}

/// Crop `[c][hp][wp]` back to `[c][h][w]`.
pub fn crop(x: &[f32], c: usize, h: usize, w: usize, hp: usize, wp: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * h * w];
    for ch in 0..c {
        for y in 0..h {
            out[(ch * h + y) * w..(ch * h + y) * w + w]
                .copy_from_slice(&x[(ch * hp + y) * wp..(ch * hp + y) * wp + w]);
        }
    }
    out
}

/// The minimum edge length this engine runs, and the reason for it.
///
/// Every down stage halves the plane, so a 64-multiple in the input becomes hp at
/// the first stage, hp/2, hp/4 and hp/8 in the body. The body's windows are 8x8, so
/// hp/8 must be at least one window: hp >= 64, i.e. any input reaches it once
/// padded. Below one window the reference's own `Block` would silently downgrade
/// its shifted windows to unshifted ones (`if input_resolution <= window_size`),
/// a behaviour this engine does not implement because at these sizes the reference
/// is out of its declared domain.
pub fn check_min_size(h: usize, w: usize, win: usize) -> Result<(), String> {
    let hp = h.div_ceil(PAD_TO) * PAD_TO;
    let wp = w.div_ceil(PAD_TO) * PAD_TO;
    if hp < 8 * win || wp < 8 * win {
        return Err(format!(
            "{h}x{w} pads to {hp}x{wp}, but the 1/8-resolution body needs at least \
             {win}x{win} pixels there - i.e. {0}x{0} before padding. The reference \
             downgrades its shifted windows to unshifted ones below that, which this \
             engine does not implement.",
            8 * win
        ));
    }
    Ok(())
}

