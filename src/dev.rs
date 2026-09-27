//! The development switches, and the refusal that keeps them out of a release build.
//!
//! Every name below selects a kernel variant or an instrument that exists so that a
//! measurement can be reproduced. None of them changes what the engine COMPUTES: the
//! default of each is the kernel the released binary uses.
//!
//! A build without the `dev` feature does not ignore them, it REFUSES them by name -
//! a script that asked for a kernel A/B must not carry on and quietly report two
//! identical runs. Nothing in `tests/` sets any of them, so the refusal costs the
//! suite nothing.
//!
//! `SCUNET_SKIP` is deliberately NOT on the list. Where the three stage skips live is a
//! real memory policy on a card this engine shares, and `tests/` runs the whole suite
//! under all three of its modes.

/// The switches a build without `dev` will not accept.
pub const DEV_SWITCHES: &[&str] = &[
    "SCUNET_PROFILE",
    "SCUNET_MEMTRACE",
    "SCUNET_GEMM",
    "SCUNET_1X1",
    "SCUNET_ATTN",
    "SCUNET_LN",
    "SCUNET_C2X2",
];

/// The name of a set development switch this build cannot honour, if there is one.
pub fn refused_switch() -> Option<&'static str> {
    if cfg!(feature = "dev") {
        return None;
    }
    DEV_SWITCHES.iter().copied().find(|n| std::env::var_os(n).is_some())
}

/// Read one of the switches above. `None` in a build without `dev`, which is also why
/// the kernel choices at the call sites collapse to their defaults.
pub fn switch(name: &str) -> Option<String> {
    if cfg!(feature = "dev") {
        std::env::var(name).ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_skip_policy_is_not_a_development_switch() {
        // It is a memory policy: a user with a small card may need `host`.
        assert!(!DEV_SWITCHES.contains(&"SCUNET_SKIP"));
    }

    #[test]
    fn the_refusal_follows_the_feature() {
        // Nothing in the environment of a test run sets one of these, so the only
        // thing being checked is the direction: a dev build refuses nothing.
        if cfg!(feature = "dev") {
            assert!(refused_switch().is_none());
        } else {
            assert!(DEV_SWITCHES.iter().all(|n| std::env::var_os(n).is_none()));
            assert!(refused_switch().is_none());
            assert!(switch("SCUNET_GEMM").is_none());
        }
    }
}
