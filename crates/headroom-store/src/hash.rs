//! Hash helpers shared by the store and the compression core.

use sha2::{Digest, Sha256};

/// CCR hash: SHA-256 of the original, truncated to 24 lowercase hex chars
/// (96 bits). Matches the Python `CompressionStore` default.
pub fn ccr_hash(original: &str) -> String {
    let digest = Sha256::digest(original.as_bytes());
    hex::encode(digest)[..24].to_string()
}

/// 12-char variant used by the native crusher paths.
pub fn ccr_hash12(original: &str) -> String {
    let digest = Sha256::digest(original.as_bytes());
    hex::encode(digest)[..12].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        // sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c...
        assert_eq!(&ccr_hash("hello")[..8], "2cf24dba");
        assert_eq!(ccr_hash("hello").len(), 24);
        assert_eq!(ccr_hash12("hello").len(), 12);
    }
}
