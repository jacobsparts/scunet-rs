//! PNG in, PNG out: 8-bit only, no scaling, no colour management.
//!
//! SCUNet's I/O contract is the simplest in the family: uint8, /255, NCHW, no mean or
//! standard-deviation normalisation, and the result is the network's output with
//! nothing added back. A grayscale checkpoint takes one channel and a colour one
//! three, and the CLI refuses a mismatch rather than guessing.
use std::path::Path;

pub struct Image {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    /// NCHW, in [0, 1].
    pub data: Vec<f32>,
}

pub fn read(path: impl AsRef<Path>) -> Result<Image, String> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let mut rdr = dec.read_info().map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; rdr.output_buffer_size()];
    let info = rdr.next_frame(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let (c, src) = match info.color_type {
        png::ColorType::Rgb => (3usize, 3usize),
        png::ColorType::Grayscale => (1, 1),
        png::ColorType::Rgba => (3, 4), // alpha dropped: the model has no fourth channel
        png::ColorType::GrayscaleAlpha => (1, 2),
        other => return Err(format!("{}: unsupported colour type {other:?}", path.display())),
    };
    let mut data = vec![0.0f32; c * h * w];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let v = buf[(y * w + x) * src + ch];
                data[ch * h * w + y * w + x] = v as f32 / 255.0;
            }
        }
    }
    Ok(Image { c, h, w, data })
}

pub fn write(path: impl AsRef<Path>, data: &[f32], c: usize, h: usize, w: usize) -> Result<(), String> {
    let path = path.as_ref();
    let color = match c {
        3 => png::ColorType::Rgb,
        1 => png::ColorType::Grayscale,
        other => return Err(format!("cannot write {other} channels as PNG")),
    };
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; c * h * w];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                // Clamp to [0,1] then scale: the model can produce values outside the
                // range at the borders, and wrapping them would be a visible seam.
                let v = data[ch * h * w + y * w + x].clamp(0.0, 1.0);
                buf[(y * w + x) * c + ch] = (v * 255.0 + 0.5) as u8;
            }
        }
    }
    wr.write_image_data(&buf).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}
