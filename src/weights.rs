//! The checkpoint: an mmap'd `.safetensors` plus the architecture it describes.
//!
//! SCUNet's eight published checkpoints share every tensor name and differ only in
//! `in_nc` (the two grayscale models take 1-channel input) and in the noise level or
//! training regime recorded in the file name. The architecture is identical across
//! all of them - `config = [4,4,4,4,4,4,4]`, `dim = 64`, `head_dim = 32`, `window = 8`
//! - but the file is not trusted to say so: `tools/convert.py` writes the constants
//! into `__metadata__`, this module reads them back and then checks them against the
//! tensors that are actually present. A checkpoint from a different family, or one
//! converted with the wrong `in_nc`, fails here with a message naming the field.
//!
//! Tensors are handed out as borrowed `&[f32]` slices of the mapping: the engine never
//! copies a weight, on either backend (`Gpu` uploads them once).
use std::collections::BTreeMap;
use std::path::Path;

use lightgpu::safetensors;

/// Which of the published checkpoints this is. Determines `in_nc` and nothing else -
/// the network is the same - but it is what the caller needs to know to pick a model
/// for a grayscale image, and it is recorded in the container so that a mismatch
/// between the file name and its contents is visible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    /// Colour, Gaussian noise sigma 15.
    Color15,
    /// Colour, Gaussian noise sigma 25.
    Color25,
    /// Colour, Gaussian noise sigma 50.
    Color50,
    /// Colour, trained for real photographs (PSNR-oriented).
    ColorRealPsnr,
    /// Colour, trained for real photographs (GAN).
    ColorRealGan,
    /// Grayscale (`in_nc = 1`), Gaussian noise sigma 15.
    Gray15,
    /// Grayscale (`in_nc = 1`), Gaussian noise sigma 25.
    Gray25,
    /// Grayscale (`in_nc = 1`), Gaussian noise sigma 50.
    Gray50,
}

impl Arch {
    fn parse(s: &str) -> Result<Arch, String> {
        Ok(match s {
            "color-15" => Arch::Color15,
            "color-25" => Arch::Color25,
            "color-50" => Arch::Color50,
            "color-real-psnr" => Arch::ColorRealPsnr,
            "color-real-gan" => Arch::ColorRealGan,
            "gray-15" => Arch::Gray15,
            "gray-25" => Arch::Gray25,
            "gray-50" => Arch::Gray50,
            other => return Err(format!(
                "variant `{other}` is not one of the eight published SCUNet checkpoints \
                 (color-15, color-25, color-50, color-real-psnr, color-real-gan, gray-15, \
                 gray-25, gray-50)"
            )),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Arch::Color15 => "color-15",
            Arch::Color25 => "color-25",
            Arch::Color50 => "color-50",
            Arch::ColorRealPsnr => "color-real-psnr",
            Arch::ColorRealGan => "color-real-gan",
            Arch::Gray15 => "gray-15",
            Arch::Gray25 => "gray-25",
            Arch::Gray50 => "gray-50",
        }
    }

    /// The two grayscale checkpoints take a single input channel.
    pub fn in_nc(&self) -> usize {
        match self {
            Arch::Gray15 | Arch::Gray25 | Arch::Gray50 => 1,
            _ => 3,
        }
    }
}

pub struct Weights {
    file: safetensors::File,
    pub arch: Arch,
    pub in_nc: usize,
    /// Transformer width. `dim = 64` for every published checkpoint.
    pub dim: usize,
    /// Attention head width; `n_heads = dim / head_dim` at each block's transformer half.
    pub head_dim: usize,
    /// Attention window edge, in pixels. 8 for every published checkpoint.
    pub window: usize,
    /// ConvTransBlocks per stage: `[down1, down2, down3, body, up3, up2, up1]`.
    pub config: [usize; 7],
    /// The checkpoint's size on disk, which the memory guard adds at face value.
    pub bytes: u64,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Weights, String> {
        let path = path.as_ref();
        let file = safetensors::File::open(path).map_err(|e| {
            // `lightgpu`'s reader names the file in EVERY one of its own messages
            // (it maps the OS error itself), so prefixing unconditionally prints the
            // path twice - which is what a mistyped `-m` used to look like:
            //     scunet: /x/nope.safetensors: /x/nope.safetensors: No such file ...
            let shown = path.display().to_string();
            let msg = e.to_string();
            if msg.starts_with(&shown) {
                msg
            } else {
                format!("{shown}: {msg}")
            }
        })?;
        let mut meta: BTreeMap<String, String> = BTreeMap::new();
        for k in ["arch", "variant", "in_nc", "dim", "config", "head_dim", "window_size"] {
            let v = file.metadata_get(k).ok_or_else(|| {
                format!(
                    "{}: no `{k}` in __metadata__ - convert the checkpoint with tools/convert.py, \
                     which records the architecture; a plain state_dict cannot be checked",
                    path.display()
                )
            })?;
            meta.insert(k.to_string(), v.to_string());
        }
        if meta["arch"] != "scunet" {
            return Err(format!(
                "{}: __metadata__ says arch = {:?}, not `scunet` - this is a checkpoint for a \
                 different engine",
                path.display(),
                meta["arch"]
            ));
        }
        let num = |k: &str| -> Result<usize, String> {
            meta[k]
                .parse()
                .map_err(|e| format!("{}: `{}` = {:?}: {}", path.display(), k, meta[k], e))
        };
        let config: Vec<usize> = meta["config"]
            .split(',')
            .map(|s| s.trim().parse::<usize>().map_err(|e| format!("config: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        if config.len() != 7 {
            return Err(format!(
                "{}: config has {} entries, SCUNet's seven stages need 7",
                path.display(),
                config.len()
            ));
        }
        let arch = Arch::parse(&meta["variant"])?;
        let in_nc = num("in_nc")?;
        // The header carries in_nc AND the variant; the two must agree, because
        // converting a grayscale checkpoint with in_nc=3 produces a file that loads
        // and then quietly runs a 3-channel stem over a 1-channel image.
        if in_nc != arch.in_nc() {
            return Err(format!(
                "{}: variant {} implies in_nc {}, the header says {}",
                path.display(),
                arch.name(),
                arch.in_nc(),
                in_nc
            ));
        }
        let w = Weights {
            arch,
            in_nc,
            dim: num("dim")?,
            head_dim: num("head_dim")?,
            window: num("window_size")?,
            config: [config[0], config[1], config[2], config[3], config[4], config[5], config[6]],
            bytes: file.len() as u64,
            file,
        };
        if w.dim % w.head_dim != 0 {
            return Err(format!(
                "{}: dim {} is not a multiple of head_dim {}",
                path.display(),
                w.dim,
                w.head_dim
            ));
        }
        w.validate(path)?;
        Ok(w)
    }

    /// The transformer width at each stage, in stage order.
    ///
    /// SCUNet halves the CONV half of each block as it descends and keeps the
    /// transformer half at a constant `dim`... except that it does not: the spatial
    /// halves are `dim//2 + dim//2 = dim` at the outermost stage and `dim + dim = 2*dim`
    /// at the next, so the ConvTransBlock's two arguments are what vary. They are
    /// recovered from the tensors below rather than tabulated, because getting this
    /// wrong is a shape error at the first mismatch and reading it off the weights is
    /// the only way to be sure the Rust agrees with what was converted.
    pub fn block_dims(&self, prefix: &str) -> (usize, usize) {
        let total = self.shape(&format!("{prefix}.conv1_1.weight"))[0];
        let trans = self.shape(&format!("{prefix}.trans_block.msa.linear.weight"))[0];
        (total - trans, trans)
    }

    /// The shapes the engine will ask for. Everything here is checked at load, so a
    /// miss in `t` later is a programming error rather than bad input.
    fn validate(&self, path: &Path) -> Result<(), String> {
        let dim = self.dim;
        let ws = self.window;
        let mut want: Vec<(String, Vec<usize>)> = vec![
            ("m_head.0.weight".into(), vec![dim, self.in_nc, 3, 3]),
            ("m_tail.0.weight".into(), vec![self.in_nc, dim, 3, 3]),
        ];
        // The seven stages, each with its own channel width and (for the up stages) a
        // transposed convolution. The names are built exactly as the Rust walker and
        // tools/reference.py build them, so this list is the single place the layout
        // is written down.
        for (stage, &count) in self.config.iter().enumerate() {
            let (name, first, tail) = match stage {
                0 => ("down1", 0usize, true),
                1 => ("down2", 0, true),
                2 => ("down3", 0, true),
                3 => ("body", 0, false),
                4 => ("up3", 1, false),
                5 => ("up2", 1, false),
                6 => ("up1", 1, false),
                _ => unreachable!(),
            };
            if first == 1 {
                // The transposed convolution: [in][out][2][2], twice the channels down.
                let (c_in, c_out) = self.up_conv_channels(name);
                want.push((format!("m_{name}.0.weight"), vec![c_in, c_out, 2, 2]));
            }
            for b in first..first + count {
                let p = format!("m_{name}.{b}");
                let (conv, trans) = self.block_dims(&p);
                want.push((format!("{p}.conv1_1.weight"), vec![conv + trans, conv + trans, 1, 1]));
                want.push((format!("{p}.conv1_1.bias"), vec![conv + trans]));
                want.push((format!("{p}.conv1_2.weight"), vec![conv + trans, conv + trans, 1, 1]));
                want.push((format!("{p}.conv1_2.bias"), vec![conv + trans]));
                want.push((format!("{p}.conv_block.0.weight"), vec![conv, conv, 3, 3]));
                want.push((format!("{p}.conv_block.2.weight"), vec![conv, conv, 3, 3]));
                for k in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"] {
                    want.push((format!("{p}.trans_block.{k}"), vec![trans]));
                }
                want.push((format!("{p}.trans_block.mlp.0.weight"), vec![4 * trans, trans]));
                want.push((format!("{p}.trans_block.mlp.0.bias"), vec![4 * trans]));
                want.push((format!("{p}.trans_block.mlp.2.weight"), vec![trans, 4 * trans]));
                want.push((format!("{p}.trans_block.mlp.2.bias"), vec![trans]));
                want.push((format!("{p}.trans_block.msa.embedding_layer.weight"), vec![3 * trans, trans]));
                want.push((format!("{p}.trans_block.msa.embedding_layer.bias"), vec![3 * trans]));
                want.push((format!("{p}.trans_block.msa.linear.weight"), vec![trans, trans]));
                want.push((format!("{p}.trans_block.msa.linear.bias"), vec![trans]));
                // One relative-position table per head, [2*ws-1][2*ws-1].
                let heads = trans / self.head_dim;
                want.push((
                    format!("{p}.trans_block.msa.relative_position_params"),
                    vec![heads, 2 * ws - 1, 2 * ws - 1],
                ));
            }
            if tail {
                // The strided convolution that follows a down stage, doubling channels.
                let (c_in, c_out) = self.down_conv_channels(name);
                want.push((format!("m_{name}.{count}.weight"), vec![c_out, c_in, 2, 2]));
            }
        }
        for (name, shape) in want {
            let got = self
                .file
                .shape(&name)
                .map_err(|e| format!("{}: {name}: {e}", path.display()))?
                .to_vec();
            if got != shape {
                return Err(format!(
                    "{}: {name} is {got:?}, the {dim}-wide/{ws}-window/{}-channel architecture in \
                     the header implies {shape:?}",
                    path.display(),
                    self.in_nc,
                ));
            }
        }
        Ok(())
    }

    /// Channel counts of a down stage's trailing stride-2 convolution, read off the
    /// stage's last block. `m_down{N}` maps `c -> 2c`; the body does not downsample.
    fn down_conv_channels(&self, name: &str) -> (usize, usize) {
        let count = match name {
            "down1" => self.config[0],
            "down2" => self.config[1],
            "down3" => self.config[2],
            _ => unreachable!(),
        };
        let last = format!("m_{name}.{}", count - 1);
        let (conv, trans) = self.block_dims(&last);
        (conv + trans, 2 * (conv + trans))
    }

    /// Channel counts of an up stage's transposed convolution, read off the stage's
    /// first block: `(2c, c)` for the outermost pair and `(4c, 2c)` for the next.
    pub fn up_conv_channels(&self, name: &str) -> (usize, usize) {
        let p = format!("m_{name}.1");
        let (conv, trans) = self.block_dims(&p);
        (2 * (conv + trans), conv + trans)
    }

    #[inline]
    fn shape(&self, name: &str) -> Vec<usize> {
        self.file.shape(name).unwrap_or_else(|e| panic!("weight `{name}`: {e}")).to_vec()
    }

    /// A weight tensor. Every name this engine asks for is validated at load, so a
    /// miss here is a programming error, not bad input.
    #[inline]
    pub fn t(&self, name: &str) -> &[f32] {
        self.file.f32(name).unwrap_or_else(|e| panic!("weight `{name}`: {e}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.file.contains(name)
    }

    /// The number of multiply-adds the convolutions cost at `h`x`w` (padded to a
    /// multiple of 64), for the run header. Attention is counted separately.
    pub fn flops_conv(&self, h: usize, w: usize) -> u64 {
        let hw = (h * w) as u64;
        let d = self.dim as u64;
        let mut f = hw * self.in_nc as u64 * d * 9; // m_head
        // Each ConvTransBlock: 1x1 expand, 3x3 + 3x3 on the conv half, 1x1 project,
        // 1x1 + 1x1 on the transformer half (counted here), plus the MLP below.
        for (stage, &count) in self.config.iter().enumerate() {
            let scale = match stage {
                0 | 6 => 1u64, // down1 / up1: full resolution
                1 | 5 => 2,
                2 | 4 => 4,
                _ => 8, // body
            };
            let sub = hw / (scale * scale);
            for b in 0..count {
                let idx = match stage {
                    4..=6 => b + 1,
                    _ => b,
                };
                let name = ["down1", "down2", "down3", "body", "up3", "up2", "up1"][stage];
                let (conv, trans) = self.block_dims(&format!("m_{name}.{idx}"));
                let (conv, trans) = (conv as u64, trans as u64);
                let c = conv + trans;
                f += sub * c * c; // conv1_1
                f += sub * conv * conv * 9 * 2; // conv_block's two 3x3
                f += sub * c * c; // conv1_2
                // Transformer half: qkv, attention (separate), proj, MLP.
                f += sub * trans * trans * 3;
                f += sub * trans * trans;
                f += 2 * sub * trans * 4 * trans;
            }
        }
        f + hw * d * self.in_nc as u64 * 9 // m_tail
    }

    /// Multiply-adds in window attention: `q.k^T` and `attn.v` per block.
    pub fn flops_attention(&self, h: usize, w: usize) -> u64 {
        let hp = h.div_ceil(64) * 64;
        let wp = w.div_ceil(64) * 64;
        let mut total = 0u64;
        for (stage, &count) in self.config.iter().enumerate() {
            let scale = match stage {
                0 | 6 => 1usize,
                1 | 5 => 2,
                2 | 4 => 4,
                _ => 8,
            };
            let (sh, sw) = (hp / scale, wp / scale);
            let windows = (sh / self.window) * (sw / self.window);
            let np = (self.window * self.window) as u64;
            for b in 0..count {
                let idx = if stage >= 4 { b + 1 } else { b };
                let name = ["down1", "down2", "down3", "body", "up3", "up2", "up1"][stage];
                let (_, trans) = self.block_dims(&format!("m_{name}.{idx}"));
                let c = trans as u64;
                total += windows as u64 * (np * np * c * 2 + np * np * c);
            }
        }
        total
    }
}
