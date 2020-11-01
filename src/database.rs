use arrayvec::ArrayString;
use sha1::{Digest, Sha1};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{Error, ErrorKind, Write};
use std::path::{Path, PathBuf};

trait ExampleDatabase {
    fn get(&self, name: &str) -> Option<Vec<u8>>;

    fn set(&mut self, name: &str, value: &[u8]);

    fn delete(&mut self, name: &str);
}

#[derive(Debug)]
/// Use a directory to store Hypothesis examples as files.
pub struct DirectoryBasedExampleDatabase {
    /// Path to the examples database.
    pub path: PathBuf,
}

impl DirectoryBasedExampleDatabase {
    /// Create a new example database that stores examples as files.
    pub fn new<P: AsRef<Path>>(path: P) -> DirectoryBasedExampleDatabase {
        let path_ref = path.as_ref();
        let _ = fs::create_dir(path_ref).map_err(ignore_error(ErrorKind::AlreadyExists));
        DirectoryBasedExampleDatabase {
            path: path_ref.to_path_buf(),
        }
    }

    #[inline]
    fn make_path(&self, key: &str) -> PathBuf {
        let mut buffer = ArrayString::<[_; 10]>::new();
        buffer
            .write_fmt(format_args!("{:.10x}", Sha1::digest(key.as_bytes())))
            .expect("10 symbols digest slice should fit 10 symbols in the buffer");
        self.path.join(buffer.as_str())
    }
}

impl ExampleDatabase for DirectoryBasedExampleDatabase {
    fn get(&self, name: &str) -> Option<Vec<u8>> {
        fs::read(self.make_path(name)).ok()
    }

    fn set(&mut self, name: &str, value: &[u8]) {
        let key_path = self.make_path(name);
        let mut target = fs::File::create(&key_path).expect("Can't create a file");
        target.write_all(value).expect("Write error");
        target.sync_all().expect("Can't sync data to disk");
    }

    fn delete(&mut self, name: &str) {
        let _ = fs::remove_file(self.make_path(name)).map_err(ignore_error(ErrorKind::NotFound));
    }
}

#[inline]
fn ignore_error(to_ignore: ErrorKind) -> impl Fn(Error) {
    move |error: Error| {
        let kind = error.kind();
        if kind != to_ignore {
            panic!("IO Error: {:?}", kind);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: [u8; 3] = [1, 2, 3];

    #[test]
    fn test_set_get() {
        let mut db = DirectoryBasedExampleDatabase::new(".minithesis-db");
        db.set("foo", &SAMPLE);
        assert_eq!(db.get("foo"), Some(SAMPLE.to_vec()));
    }

    #[test]
    fn test_set_delete_get() {
        let mut db = DirectoryBasedExampleDatabase::new(".minithesis-db");
        db.set("bar", &SAMPLE);
        db.delete("bar");
        assert!(db.get("bar").is_none());
    }
}
