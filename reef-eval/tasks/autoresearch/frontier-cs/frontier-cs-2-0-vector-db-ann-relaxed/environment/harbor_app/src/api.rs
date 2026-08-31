use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct InsertRequest {
    pub id: u64,
    pub vector: Vec<f32>,
}

#[derive(Serialize)]
pub struct InsertResponse {
    pub status: String,
}

#[derive(Deserialize)]
pub struct BulkInsertItem {
    pub id: u64,
    pub vector: Vec<f32>,
}

#[derive(Deserialize)]
pub struct BulkInsertRequest {
    pub vectors: Vec<BulkInsertItem>,
}

#[derive(Serialize)]
pub struct BulkInsertResponse {
    pub status: String,
    pub inserted: usize,
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
