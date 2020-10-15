use arrayvec::ArrayString;
use sha1::{Digest, Sha1};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

trait ExampleDatabase {
    fn get(&self, name: &str) -> Option<Vec<u8>>;

    fn set(&self, name: &str, value: &[u8]);

    fn delete(&self, name: &str);
}

type Cache = HashMap<String, ArrayString<[u8; 10]>>;

#[derive(Debug)]
/// Use a directory to store Hypothesis examples as files.
pub struct DirectoryBasedExampleDatabase {
    /// Path to the examples database.
    pub path: PathBuf,
    cache: RefCell<Cache>,
}

macro_rules! hash {
    ($var:ident) => {{
        // Since the hash size is known statically, there is not need to allocate a `String` here
        // via using the `format!` macro. Instead we allocate on the stack
        // On average it gives ~10% performance improvement for calculating hash strings
        let mut out = ArrayString::<[_; 10]>::new();
        out.write_fmt(format_args!("{:.10x}", Sha1::digest($var.as_bytes())))
            .expect("Hash is always representable in hex format");
        out
    }};
}

impl DirectoryBasedExampleDatabase {
    /// Create a new example database that stores examples as files.
    pub fn new<P: AsRef<Path>>(path: P) -> DirectoryBasedExampleDatabase {
        DirectoryBasedExampleDatabase {
            path: path.as_ref().to_path_buf(),
            cache: RefCell::new(HashMap::new()),
        }
    }

    #[inline]
    fn make_path(&self, key: &str) -> PathBuf {
        let mut cache = self.cache.borrow_mut();
        if let Some(hashed) = cache.get(key) {
            self.path.join(hashed.as_str())
        } else {
            let hashed = hash!(key);
            cache.insert(key.to_string(), hashed);
            self.path.join(hashed.as_str())
        }
    }
}

impl ExampleDatabase for DirectoryBasedExampleDatabase {
    fn get(&self, name: &str) -> Option<Vec<u8>> {
        let path = self.make_path(name);
        fs::read(path).ok()
    }

    fn set(&self, name: &str, value: &[u8]) {
        let key_path = self.make_path(name);
        let mut target = fs::File::create(&key_path).expect("Can't create a file");
        target.write_all(value).expect("Write error");
        target.sync_all().expect("Can't sync data to disk");
    }

    fn delete(&self, name: &str) {
        let path = self.make_path(name);
        let _ = fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: [u8; 3] = [1, 2, 3];

    #[test]
    fn test_set_get() {
        let db = DirectoryBasedExampleDatabase::new(".minithesis-db");
        db.set("foo", &SAMPLE);
        assert_eq!(db.get("foo"), Some(SAMPLE.to_vec()));
    }

    #[test]
    fn test_set_delete_get() {
        let db = DirectoryBasedExampleDatabase::new(".minithesis-db");
        db.set("bar", &SAMPLE);
        db.delete("bar");
        assert!(db.get("bar").is_none());
    }
}
