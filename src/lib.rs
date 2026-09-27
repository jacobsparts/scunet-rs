//! SCUNet (Swin-Conv-UNet) image restoration, CPU and CUDA backends.
//!
//! See `src/weights.rs` for what the checkpoint must contain and what is checked at
//! load time, `src/plan.rs` for the padding geometry (replicate to a multiple of 64,
//! run once, crop - no tiling), `src/backend.rs` for the trait the CLI shares, and
//! `tools/reference.py` for the torch-free reference the backends are validated
//! against.
pub mod backend;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod cpu;
pub mod dev;
pub mod fixture;
pub mod image;
pub mod memguard;
pub mod plan;
pub mod weights;

pub use backend::Backend;
pub use weights::{Arch, Weights};
