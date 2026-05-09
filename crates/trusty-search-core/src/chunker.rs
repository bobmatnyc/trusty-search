use serde::{Deserialize, Serialize};

/// A code chunk extracted from a source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawChunk {
    /// Collision-safe ID: "{path}:{start_line}:{end_line}"
    pub id: String,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub function_name: Option<String>,
    pub language: Option<String>,
}

/// Overlapping sliding window chunker.
/// window=150 lines, stride=50 lines (67% overlap) for high recall on config/multi-function files.
pub fn chunk_text(file: &str, content: &str, window: usize, stride: usize) -> Vec<RawChunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < lines.len() {
        let end = (start + window).min(lines.len());
        let text = lines[start..end].join("\n");
        chunks.push(RawChunk {
            id: format!("{}:{}:{}", file, start + 1, end),
            file: file.to_string(),
            start_line: start + 1,
            end_line: end,
            content: text,
            function_name: None,
            language: None,
        });
        if end == lines.len() { break; }
        start += stride;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlapping_chunks() {
        let content = (1..=200).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let chunks = chunk_text("test.rs", &content, 150, 50);
        assert!(chunks.len() >= 2, "should produce multiple overlapping chunks");
        // Verify overlap: chunk[1] starts at stride offset
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[1].start_line, 51);
    }

    #[test]
    fn test_chunk_id_format() {
        let chunks = chunk_text("src/main.rs", "line1\nline2\nline3", 150, 50);
        assert!(chunks[0].id.starts_with("src/main.rs:"));
    }
}
