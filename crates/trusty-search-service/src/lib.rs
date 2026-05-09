pub mod client;
pub mod indexed_files;
pub mod server;
pub mod watch_loop;
pub mod watcher;

pub use indexed_files::IndexedFiles;
pub use server::SearchAppState;
pub use watch_loop::{spawn_watch_loop, WatcherTask};
pub use watcher::{FileWatcher, WatchEvent};
