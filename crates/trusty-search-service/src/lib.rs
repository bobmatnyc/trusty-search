pub mod client;
pub mod daemon;
pub mod indexed_files;
pub mod server;
pub mod watch_loop;
pub mod watcher;

pub use daemon::{daemon_lock_path, daemon_port_path, run_daemon, DaemonError, DaemonHandle};
pub use indexed_files::IndexedFiles;
pub use server::SearchAppState;
pub use watch_loop::{spawn_watch_loop, WatcherTask};
pub use watcher::{FileWatcher, WatchEvent};
