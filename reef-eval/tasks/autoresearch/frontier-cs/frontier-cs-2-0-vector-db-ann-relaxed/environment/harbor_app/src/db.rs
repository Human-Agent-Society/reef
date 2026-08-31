use crate::api::SearchResult;

pub struct VectorDB {
    // Implement your index here.
}

impl VectorDB {
    pub fn new() -> Self {
        todo!("initialize your vector database")
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        todo!("insert one vector")
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        todo!("insert a batch of vectors")
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        todo!("return approximate nearest neighbors")
    }
}
