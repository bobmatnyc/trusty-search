pub mod client;
pub mod config;
pub mod daemon;
pub mod indexed_files;
pub mod persistence;
pub mod persistence_loader;
pub mod reindex;
pub mod server;
pub mod ui;
pub mod walker;
pub mod watch_loop;
pub mod watcher;

pub use config::{load_user_config, LoadedUserConfig};
pub use daemon::{
    daemon_env_path, daemon_lock_path, daemon_port_path, http_addr_path, is_already_running,
    load_daemon_env, run_daemon, running_daemon_pid, save_daemon_env, DaemonError, DaemonHandle,
    PERSISTED_ENV_VARS,
};
pub use indexed_files::IndexedFiles;
pub use server::SearchAppState;
pub use watch_loop::{spawn_watch_loop, WatcherTask};
pub use watcher::{FileWatcher, WatchEvent};
