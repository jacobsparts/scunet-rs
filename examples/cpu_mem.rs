//! WHAT THE CPU PATH'S MEMORY ACTUALLY IS.
//!
//! At 1024x1024 the CPU forward holds 3052 MiB live - 2.84 KB per pixel - against
//! the CUDA path's 1132 MiB (1.24 KB/px) for the same model and the same image. This
//! names every buffer responsible, rather than inferring them from a reading of
//! `cpu::forward`.
//!
//! HOW. A global allocator that counts live and peak bytes, plus a table of the
//! allocations over 1 MiB - large enough to be a plane, small enough to be cheap to
//! track - which is dumped whenever a new peak is set. So the output is the live set
//! AT ITS WORST MOMENT, largest first, which is the question.
//!
//!     cargo run --release --features cuda --example cpu_mem -- [size]
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TOTAL: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);

/// Allocations of at least this size are tracked in the table below.
const BIG: usize = 1 << 20;
const SLOTS: usize = 4096;
static mut PTRS: [usize; SLOTS] = [0; SLOTS];
static mut SIZES: [usize; SLOTS] = [0; SLOTS];
static LOCK: AtomicBool = AtomicBool::new(false);
/// The live total when the snapshot below was taken, and the snapshot itself.
static SNAP_AT: AtomicUsize = AtomicUsize::new(0);
static mut SNAP: [(usize, usize); 12] = [(0, 0); 12];

fn lock() {
    while LOCK.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        std::hint::spin_loop();
    }
}
fn unlock() {
    LOCK.store(false, Ordering::Release);
}

/// Hash a pointer to a slot, with linear probing. No allocation: this runs inside
/// the allocator, so anything that allocates would recurse.
unsafe fn table_insert(p: usize, n: usize) {
    let mut i = (p >> 6) % SLOTS;
    for _ in 0..SLOTS {
        let ptrs = unsafe { &mut *std::ptr::addr_of_mut!(PTRS) };
        if ptrs[i] == 0 {
            ptrs[i] = p;
            let sizes = unsafe { &mut *std::ptr::addr_of_mut!(SIZES) };
            sizes[i] = n;
            return;
        }
        i = (i + 1) % SLOTS;
    }
}

unsafe fn table_remove(p: usize) {
    let mut i = (p >> 6) % SLOTS;
    for _ in 0..SLOTS {
        let ptrs = unsafe { &mut *std::ptr::addr_of_mut!(PTRS) };
        if ptrs[i] == p {
            ptrs[i] = 0;
            return;
        }
        if ptrs[i] == 0 {
            return;
        }
        i = (i + 1) % SLOTS;
    }
}

/// Snapshot the largest live allocations, when a new peak is set.
unsafe fn snapshot(peak: usize) {
    let ptrs = unsafe { &*std::ptr::addr_of!(PTRS) };
    let sizes = unsafe { &*std::ptr::addr_of!(SIZES) };
    let mut top: [(usize, usize); 12] = [(0, 0); 12];
    for i in 0..SLOTS {
        let n = sizes[i];
        if n == 0 {
            continue;
        }
        for j in 0..12 {
            if n > top[j].1 {
                for k in (j + 1..12).rev() {
                    top[k] = top[k - 1];
                }
                top[j] = (ptrs[i], n);
                break;
            }
        }
    }
    let snap = unsafe { &mut *std::ptr::addr_of_mut!(SNAP) };
    *snap = top;
    SNAP_AT.store(peak, Ordering::Relaxed);
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            TOTAL.fetch_add(l.size(), Ordering::Relaxed);
            COUNT.fetch_add(1, Ordering::Relaxed);
            if now > PEAK.load(Ordering::Relaxed) {
                PEAK.store(now, Ordering::Relaxed);
                if l.size() >= BIG {
                    lock();
                    unsafe { table_insert(p as usize, l.size()) };
                    if now > SNAP_AT.load(Ordering::Relaxed) + (1 << 20) {
                        unsafe { snapshot(now) };
                    }
                    unlock();
                }
            } else if l.size() >= BIG {
                lock();
                unsafe { table_insert(p as usize, l.size()) };
                unlock();
            }
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        if l.size() >= BIG {
            lock();
            unsafe { table_remove(p as usize) };
            unlock();
        }
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
            let now = LIVE.fetch_add(new, Ordering::Relaxed) + new;
            TOTAL.fetch_add(new.saturating_sub(l.size()), Ordering::Relaxed);
            COUNT.fetch_add(1, Ordering::Relaxed);
            if l.size() >= BIG || new >= BIG {
                lock();
                unsafe { table_remove(p as usize) };
                if new >= BIG {
                    unsafe { table_insert(q as usize, new) };
                }
                unlock();
            }
            if now > PEAK.load(Ordering::Relaxed) {
                PEAK.store(now, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static A: Counting = Counting;

use scunet::plan::Plan;
use scunet::weights::Weights;

fn mib(b: usize) -> f64 {
    b as f64 / 1048576.0
}

fn main() -> Result<(), String> {
    let size: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let p = Plan::new(size, size, wt.window);
    let (hp, wp) = (p.hp, p.wp);
    let x = vec![0.5f32; wt.in_nc * hp * wp];
    let mut be = scunet::cpu::Cpu::new(&wt)?;
    let _ = scunet::backend::run(&mut be, &x, wt.in_nc, size, size, &wt)?;

    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    TOTAL.store(0, Ordering::Relaxed);
    COUNT.store(0, Ordering::Relaxed);
    SNAP_AT.store(0, Ordering::Relaxed);
    let y = scunet::backend::run(&mut be, &x, wt.in_nc, size, size, &wt)?;

    let peak = PEAK.load(Ordering::Relaxed);
    let total = TOTAL.load(Ordering::Relaxed);
    println!("CPU ALLOCATION at {size}x{size} (padded {hp}x{wp}), one forward after a warmup");
    println!("  PEAK LIVE   {:>8.1} MiB   = {:.2} KB per input pixel", mib(peak), peak as f64 / (size * size) as f64 / 1024.0);
    println!("  TOTAL ASKED {:>8.1} MiB   over {} allocation calls ({:.1}x the peak)", mib(total), COUNT.load(Ordering::Relaxed), total as f64 / peak.max(1) as f64);
    println!("  live at end {:>8.1} MiB   (the returned output is {:.1} MiB)", mib(LIVE.load(Ordering::Relaxed)), mib(y.len() * 4));
    println!("\n  THE LIVE SET AT ITS WORST MOMENT (allocations of 1 MiB or more, largest first):");
    let snap = unsafe { &*std::ptr::addr_of!(SNAP) };
    let mut sum = 0usize;
    for (i, (_, n)) in snap.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        sum += n;
        println!("    {:>2}. {:>9.1} MiB", i + 1, mib(*n));
    }
    println!("    the twelve largest sum to {:.1} MiB of the {:.1} MiB peak", mib(sum), mib(peak));
    drop(y);
    Ok(())
}
