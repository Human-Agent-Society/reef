use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct LoadRequest {
    pub index_path: Option<String>,
    pub graph_path: Option<String>,
    pub vector_path: String,
    pub vector_dtype: Option<String>,
    pub pq_vector_path: Option<String>,
    pub pq_compressed_path: Option<String>,
    pub pq_pivots_path: Option<String>,
}

#[derive(Serialize)]
pub struct LoadResponse {
    pub status: String,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub top_k: u32,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f64,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

impl LoadRequest {
    pub fn index_path(&self) -> &str {
        self.index_path
            .as_deref()
            .or(self.graph_path.as_deref())
            .expect("load request must include index_path or graph_path")
    }

    pub fn pq_compressed_path(&self) -> Option<&str> {
        self.pq_compressed_path
            .as_deref()
            .or(self.pq_vector_path.as_deref())
    }
}
