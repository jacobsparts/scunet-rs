//! Development: dump every stage of the walker for one fixture, so a divergence can be
//! located by STAGE rather than by bisection on the final image.
use std::fs;

use scunet::cpu;
use scunet::Weights;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fixture = args.first().cloned().unwrap_or_else(|| "color_real_psnr_64.bin".into());
    let outdir = args.get(1).cloned().unwrap_or_else(|| "/tmp/rustdump".into());
    fs::create_dir_all(&outdir).unwrap();
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors").unwrap();
    let f = scunet::fixture::Fixture::load(format!("tests/data/{fixture}")).unwrap();
    let mut shapes = Vec::new();
    let y = cpu::forward_dump(&wt, &f.input, f.h, f.w, &mut |name, plane| {
        let p = format!("{outdir}/{name}.bin");
        let mut bytes = Vec::with_capacity(plane.len() * 4);
        for v in plane {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&p, &bytes).unwrap();
        shapes.push(format!("{name} {} {} {}", plane.len(), f.h, f.w));
    })
    .unwrap();
    println!("{} stages written to {outdir}", shapes.len());
    println!("final: {}", f.compare(&y).0);
    // The stage shapes, so the numpy side knows what to expect without re-deriving them.
    for s in &shapes {
        println!("{s}");
    }
}
