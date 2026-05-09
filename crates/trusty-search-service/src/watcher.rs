/// FileWatcher: notify-debouncer-mini, 500ms debounce.
/// Sends changed paths to a tokio channel for the indexer to process.
pub struct FileWatcher;

impl FileWatcher {
    pub async fn spawn(_root: &std::path::Path) -> anyhow::Result<tokio::sync::mpsc::Receiver<std::path::PathBuf>> {
        let (_tx, rx) = tokio::sync::mpsc::channel(100);
        // TODO: implement with notify-debouncer-mini
        Ok(rx)
    }
}
