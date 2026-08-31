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
        .route("/insert", post(handle_insert))
        .route("/bulk_insert", post(handle_bulk_insert))
        .route("/search", post(handle_search))
        .with_state(db);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_insert(
    State(db): State<Arc<VectorDB>>,
    Json(req): Json<InsertRequest>,
) -> Json<InsertResponse> {
    db.insert(req.id, req.vector);
    Json(InsertResponse {
        status: "ok".to_string(),
    })
}

async fn handle_bulk_insert(
    State(db): State<Arc<VectorDB>>,
    Json(req): Json<BulkInsertRequest>,
) -> Json<BulkInsertResponse> {
    let vectors = req.vectors.into_iter().map(|v| (v.id, v.vector)).collect();
    let inserted = db.bulk_insert(vectors);
    Json(BulkInsertResponse {
        status: "ok".to_string(),
        inserted,
    })
}

async fn handle_search(
    State(db): State<Arc<VectorDB>>,
    Json(req): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let results = db.search(&req.vector, req.top_k);
    Json(SearchResponse { results })
}
