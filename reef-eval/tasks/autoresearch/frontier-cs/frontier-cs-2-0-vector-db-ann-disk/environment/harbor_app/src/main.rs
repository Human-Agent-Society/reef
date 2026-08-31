use axum::{extract::State, routing::post, Json, Router};
use std::sync::Arc;
use tokio::net::TcpListener;

mod api;
mod db;
mod distance;

use api::*;
use db::VectorDB;

#[tokio::main]
async fn main() {
    let db = Arc::new(VectorDB::new());
    let app = Router::new()
        .route("/load", post(handle_load))
        .route("/search", post(handle_search))
        .with_state(db);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_load(
    State(db): State<Arc<VectorDB>>,
    Json(req): Json<LoadRequest>,
) -> Json<LoadResponse> {
    db.load(
        req.index_path(),
        &req.vector_path,
        req.vector_dtype.as_deref(),
        req.pq_compressed_path(),
        req.pq_pivots_path.as_deref(),
    );
    Json(LoadResponse {
        status: "ok".to_string(),
    })
}

async fn handle_search(
    State(db): State<Arc<VectorDB>>,
    Json(req): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let results = db.search(&req.vector, req.top_k);
    Json(SearchResponse { results })
}
