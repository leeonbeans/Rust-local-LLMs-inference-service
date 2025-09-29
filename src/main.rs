use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::VecDeque, net::SocketAddr, sync::Arc, time::Duration};
use axum::routing::get_service;
use tokio::sync::Mutex;
use sysinfo::{Networks, System, Disks};
use tower_http::{
    cors::{CorsLayer, Any},
    services::ServeFile,
};
use sqlx::{Pool, Sqlite, SqlitePool, Row};
use tracing_subscriber;

const SAMPLE_CAPACITY: usize = 60;

#[derive(Clone)]
struct AppState {
    client: Client,
    llama_base: String,
    metrics: Arc<Metrics>,
    db: Pool<Sqlite>, // 新增 SQLite 连接池
}

struct Metrics {
    cpu: Mutex<VecDeque<f32>>,
    memory: Mutex<VecDeque<f32>>,
    disk: Mutex<VecDeque<f32>>,
    network: Mutex<VecDeque<f32>>,
    prev_net_total: Mutex<u128>,
}

impl Metrics {
    fn new() -> Self {
        Self {
            cpu: Mutex::new(VecDeque::with_capacity(SAMPLE_CAPACITY)),
            memory: Mutex::new(VecDeque::with_capacity(SAMPLE_CAPACITY)),
            disk: Mutex::new(VecDeque::with_capacity(SAMPLE_CAPACITY)),
            network: Mutex::new(VecDeque::with_capacity(SAMPLE_CAPACITY)),
            prev_net_total: Mutex::new(0),
        }
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    model: String,
    temperature: f32,
    top_p: f32,
    reset: Option<bool>, // 新增，支持清空对话
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    llama: String,
    message: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // 初始化 SQLite
    let db = SqlitePool::connect("sqlite://chat.db?mode=rwc").await.unwrap();
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        "#
    ).execute(&db).await.unwrap();

    let llama_base =
        std::env::var("LLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let metrics = Arc::new(Metrics::new());
    spawn_metrics_collector(metrics.clone());

    let state = AppState {
        client,
        llama_base,
        metrics,
        db,
    };

    // 静态文件服务
    let static_files = ServeFile::new("./static/index.html");

    // 构建路由
    let app = Router::new()
        .route("/", get_service(static_files))
        .route("/api/health", get(health_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/system_stats", get(system_stats_handler))
        .route("/api/history", get(history_handler)) // 新增历史接口
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    // 绑定并启动
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3002").await.unwrap();
    println!("🚀 Rust LLM Service running at http://127.0.0.1:3002");
    axum::serve(listener, app).await.unwrap();
}

// ---------------- Handlers ----------------

// 健康检查
async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    let llama_health = check_llama_health(&state.client, &state.llama_base).await;
    let (llama, msg) = match llama_health {
        Ok(true) => ("ok".to_string(), "Backend + Llama connected".to_string()),
        Ok(false) => ("loading_or_error".to_string(), "Backend ok, llama not ready".to_string()),
        Err(e) => ("unreachable".to_string(), format!("Backend ok, llama unreachable: {}", e)),
    };
    let res = HealthResponse {
        status: "ok".to_string(),
        llama,
        message: msg,
    };
    (StatusCode::OK, Json(res))
}

async fn check_llama_health(client: &Client, base: &str) -> Result<bool, String> {
    let url = format!("{}/health", base.trim_end_matches('/'));
    match client.get(&url).send().await {
        Ok(resp) => Ok(resp.status().is_success()),
        Err(e) => Err(format!("{:?}", e)),
    }
}

// 聊天（带上下文 + SQLite）
async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    if req.reset.unwrap_or(false) {
        sqlx::query("DELETE FROM chat_history").execute(&state.db).await.ok();
        return (StatusCode::OK, Json(json!({"response": "[历史已清空]"})));
    }

    // 保存用户消息
    sqlx::query("INSERT INTO chat_history (role, content) VALUES (?, ?)")
        .bind("user")
        .bind(&req.message)
        .execute(&state.db)
        .await
        .ok();

    // 获取最近 20 条历史
    let rows = sqlx::query("SELECT role, content FROM chat_history ORDER BY id DESC LIMIT 20")
        .fetch_all(&state.db)
        .await
        .unwrap();

    let history: Vec<Value> = rows.into_iter()
        .rev()
        .map(|r| {
            let role: String = r.get("role");
            let content: String = r.get("content");
            json!({"role": role, "content": content})
        })
        .collect();

    let target = format!("{}/v1/chat/completions", state.llama_base.trim_end_matches('/'));
    let payload = json!({
        "model": req.model,
        "messages": history,
        "temperature": req.temperature,
        "top_p": req.top_p
    });

    match state.client.post(&target).json(&payload).send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("llama returned {}", resp.status())})),
                );
            }

            let v: Value = resp.json().await.unwrap_or(json!({}));
            let reply = v["choices"]
                .get(0)
                .and_then(|c| c["message"]["content"].as_str())
                .unwrap_or("[无返回]")
                .to_string();

            // 保存助手回复
            sqlx::query("INSERT INTO chat_history (role, content) VALUES (?, ?)")
                .bind("assistant")
                .bind(&reply)
                .execute(&state.db)
                .await
                .ok();

            (StatusCode::OK, Json(json!({ "response": reply })))
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": format!("cannot reach llama-server: {}", e)})),
        ),
    }
}

// 历史查询
async fn history_handler(State(state): State<AppState>) -> impl IntoResponse {
    // 查询最近 20 条（按 id 倒序，最后再反转成正序，保证显示从旧到新）
    let rows = sqlx::query("SELECT role, content FROM chat_history ORDER BY id DESC LIMIT 20")
        .fetch_all(&state.db)
        .await
        .unwrap_or_default();

    let mut history: Vec<serde_json::Value> = rows
        .into_iter()
        .rev()
        .map(|r| {
            let role: String = r.get("role");
            let content: String = r.get("content");
            json!({ "role": role, "content": content })
        })
        .collect();

    (StatusCode::OK, Json(history))
}

// 系统监控
async fn system_stats_handler(State(state): State<AppState>) -> impl IntoResponse {
    let cpu = state.metrics.cpu.lock().await.clone().into_iter().collect::<Vec<_>>();
    let memory = state.metrics.memory.lock().await.clone().into_iter().collect::<Vec<_>>();
    let disk = state.metrics.disk.lock().await.clone().into_iter().collect::<Vec<_>>();
    let network = state.metrics.network.lock().await.clone().into_iter().collect::<Vec<_>>();

    let body = json!({
        "cpu": cpu,
        "memory": memory,
        "disk": disk,
        "network": network,
    });
    (StatusCode::OK, Json(body))
}

// ---------------- Metrics ----------------

fn spawn_metrics_collector(metrics: Arc<Metrics>) {
    tokio::spawn(async move {
        let mut sys = System::new_all();
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        let mut net = Networks::new();
        let mut disks = Disks::new();
        loop {
            interval.tick().await;
            sys.refresh_cpu_usage();
            sys.refresh_memory();
            disks.refresh(false);
            net.refresh(false);

            let cpu_pct = sys.global_cpu_usage();

            let total_mem = sys.total_memory() as f32;
            let used_mem = (sys.total_memory() - sys.available_memory()) as f32;
            let mem_pct = if total_mem > 0.0 { used_mem / total_mem * 100.0 } else { 0.0 };

            let mut total_space: u128 = 0;
            let mut avail_space: u128 = 0;
            for disk in disks.iter() {
                total_space += disk.total_space() as u128;
                avail_space += disk.available_space() as u128;
            }
            let disk_pct = if total_space > 0 {
                (total_space - avail_space) as f32 / total_space as f32 * 100.0
            } else {
                0.0
            };

            let mut net_total: u128 = 0;
            for (_name, data) in net.list() {
                net_total += (data.received() as u128) + (data.transmitted() as u128);
            }
            let mut prev = metrics.prev_net_total.lock().await;
            let delta = net_total.saturating_sub(*prev);
            *prev = net_total;
            drop(prev);

            let bytes_per_sec = (delta as f64) / 5.0;
            let mbps = (bytes_per_sec * 8.0) / (1024.0 * 1024.0);
            let net_pct = ((mbps / 1000.0) * 100.0).min(100.0) as f32;

            push(&metrics.cpu, cpu_pct).await;
            push(&metrics.memory, mem_pct).await;
            push(&metrics.disk, disk_pct).await;
            push(&metrics.network, net_pct).await;
        }
    });
}

async fn push(m: &Mutex<VecDeque<f32>>, val: f32) {
    let mut dq = m.lock().await;
    if dq.len() >= SAMPLE_CAPACITY {
        dq.pop_front();
    }
    dq.push_back((val * 100.0).round() / 100.0);
}
