use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
    routing::get_service,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::VecDeque, net::SocketAddr, sync::Arc, time::Duration,
    process::{Command, Stdio, Child},
    collections::HashMap,
};
use std::io::Write;
use tokio::sync::Mutex;
use sysinfo::{Networks, System, Disks};
use tower_http::{
    cors::{CorsLayer, Any},
    services::ServeFile,
};
use sqlx::{Pool, Sqlite, SqlitePool, Row};
use tracing_subscriber;

// === Candle 相关 ===
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;
use anyhow::Result;
use candle_examples::token_output_stream::TokenOutputStream;

const SAMPLE_CAPACITY: usize = 60;

#[derive(Clone)]
struct AppState {
    client: Client,
    llama_base: String,
    metrics: Arc<Metrics>,
    db: Pool<Sqlite>,
    current_model: Arc<Mutex<Option<Child>>>,
    model_paths: Arc<HashMap<String, String>>,
    candle_models: Arc<HashMap<String, (String, String)>>,
    loaded_candle: Arc<Mutex<Option<(ModelWeights, Tokenizer)>>>,
    current_model_name: Arc<Mutex<Option<String>>>,
    candle_model_loaded: Arc<Mutex<bool>>,
    llama_model_loaded: Arc<Mutex<bool>>,
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
    reset: Option<bool>,
}

#[derive(Deserialize)]
struct LoadModelRequest {
    model: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    llama: String,
    message: String,
    candle: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let mut model_paths = HashMap::new();
    model_paths.insert("qwen3".to_string(), ".\\model\\qwen2.5-0.5b.gguf".to_string());
    model_paths.insert("llama".to_string(), ".\\llama2\\llama-2-7b.gguf".to_string());

    // ✅ Candle 模型路径表
    let mut candle_models = HashMap::new();
    candle_models.insert(
        "qwen3-candle".to_string(),
        (
            ".\\model\\candle\\qwen3\\qwen3.gguf".to_string(),
            ".\\model\\candle\\qwen3\\tokenizer.json".to_string(),
        ),
    );

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
        .timeout(Duration::from_secs(300))
        .build()
        .unwrap();

    let metrics = Arc::new(Metrics::new());
    spawn_metrics_collector(metrics.clone());

    let state = AppState {
        client,
        llama_base,
        metrics,
        db,
        current_model: Arc::new(Mutex::new(None)),
        model_paths: Arc::new(model_paths),
        candle_models: Arc::new(candle_models),
        loaded_candle: Arc::new(Mutex::new(None)),
        current_model_name: Arc::new(Mutex::new(None)),
        candle_model_loaded: Arc::new(Mutex::new(false)),
        llama_model_loaded: Arc::new(Mutex::new(false)),
    };

    let static_files = ServeFile::new("./static/index.html");

    let app = Router::new()
        .route("/", get_service(static_files))
        .route("/api/health", get(health_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/system_stats", get(system_stats_handler))
        .route("/api/history", get(history_handler))
        .route("/api/load_model", post(load_model_handler))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3002").await.unwrap();
    println!("🚀 Rust LLM Service running at http://127.0.0.1:3002");
    axum::serve(listener, app).await.unwrap();
}
// ---------------- Handlers ----------------

async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    let mut status = "ok".to_string();
    let mut llama = "none".to_string();
    let mut candle = "none".to_string();
    let mut message = "No model loaded".to_string();

    // 检查 Candle 是否加载
    if *state.candle_model_loaded.lock().await {
        candle = "candle_loaded".to_string();
        message = "Candle 模型已加载".to_string();
    }

    // 检查 llama.cpp 服务
    if *state.llama_model_loaded.lock().await {
        match check_llama_health(&state.client, &state.llama_base).await {
            Ok(true) => {
                llama = "llama_loaded".to_string();
                message = "llama.cpp 模型连接正常".to_string();
            }
            Ok(false) => {
                llama = "loading_or_error".to_string();
                message = "llama.cpp 未准备好".to_string();
            }
            Err(e) => {
                llama = "unreachable".to_string();
                message = format!("无法连接 llama.cpp: {}", e);
            }
        }
    }
    let res = HealthResponse { status, llama, candle, message };
    (StatusCode::OK, Json(res))
}


async fn check_llama_health(client: &Client, base: &str) -> Result<bool, String> {
    let url = format!("{}/health", base.trim_end_matches('/'));
    match client.get(&url).send().await {
        Ok(resp) => Ok(resp.status().is_success()),
        Err(e) => Err(format!("{:?}", e)),
    }
}

// 聊天（自动识别 candle/llama）
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

    // 检测是否是 candle 模型
    if req.model.contains("candle") {
        let mut lock = state.loaded_candle.lock().await;
        let mut name = state.current_model_name.lock().await;
        *name = Some(req.model.clone());
        if let Some((model, tokenizer)) = &mut *lock {
            let response = run_candle_inference(model, tokenizer, &req.message,req.temperature as f64,Some(req.top_p as f64))
            .unwrap_or_else(|e| format!("Candle error: {e}"));
            sqlx::query("INSERT INTO chat_history (role, content) VALUES (?, ?)")
                .bind("assistant")
                .bind(&response)
                .execute(&state.db)
                .await
                .ok();
            return (StatusCode::OK, Json(json!({ "response": response })));
        } else {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Candle 模型未加载"})),
            );
        }
    }

    // llama.cpp 模型
    let rows = sqlx::query("SELECT role, content FROM chat_history ORDER BY id DESC LIMIT 20")
        .fetch_all(&state.db)
        .await
        .unwrap();

    let history: Vec<Value> = rows
        .into_iter()
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

// 加载模型（支持 llama-server 与 Candle）
async fn load_model_handler(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> impl IntoResponse {
    // 1️⃣ 如果是 Candle 模型
    if req.model.contains("candle") {
        let Some((model_path, tokenizer_path)) = state.candle_models.get(&req.model) else {
            return (StatusCode::BAD_REQUEST, Json(json!({"status": "error", "error": "未知 Candle 模型"})));
        };
        println!("🧠 正在加载 Candle 模型: {}", req.model);

        match load_candle_model(model_path, tokenizer_path) {
            Ok((model, tokenizer)) => {
                let mut guard = state.loaded_candle.lock().await;
                *guard = Some((model, tokenizer));
                *state.candle_model_loaded.lock().await = true;
                return (StatusCode::OK, Json(json!({"status": "ok", "model": req.model})));
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"status": "error", "error": e.to_string()})),
                );
            }
        }
    }

    // 2️⃣ llama.cpp 模型
    let Some(path) = state.model_paths.get(&req.model) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"status": "error", "error": "未知模型"})),
        );
    };

    // 关闭旧进程
    {
        let mut guard = state.current_model.lock().await;
        if let Some(child) = guard.as_mut() {
            #[cfg(target_os = "windows")]
            {
                let _ = Command::new("taskkill")
                    .args(&["/PID", &child.id().to_string(), "/F"])
                    .output();
            }
            let _ = child.kill();
            *guard = None;
            *state.llama_model_loaded.lock().await = false;
        }
    }

    // 启动新进程
    let exe = ".\\model\\llama\\llama-server.exe";
    match Command::new(exe)
        .args(&["-m", path])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => {
            let mut guard = state.current_model.lock().await;
            *guard = Some(child);
            *state.llama_model_loaded.lock().await = true;
            (StatusCode::OK, Json(json!({"status": "ok", "model": req.model})))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"status": "error", "error": e.to_string()})),
        ),
    }
}

// ---------------- Candle 相关函数 ----------------
fn load_candle_model(model_path: &str, tokenizer_path: &str) -> Result<(ModelWeights, Tokenizer)> {
    let device = Device::Cpu;
    let mut model_file = std::fs::File::open(model_path)?;
    let model_data = gguf_file::Content::read(&mut model_file).map_err(|e| e.with_path(model_path))?;
    let model = ModelWeights::from_gguf(model_data, &mut model_file, &device)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((model, tokenizer))
}

fn run_candle_inference(model: &mut ModelWeights, tokenizer: &Tokenizer, prompt: &str,temperature: f64, top_p: Option<f64>) -> anyhow::Result<String> {
    // 固定 prompt 模板
    let formatted_prompt = format!(
        "<|im_start|>system\n你是一个友好的 AI 助手，回答要简洁自然。\n<|im_end|>\n\
         <|im_start|>user\n{}\n<|im_end|>\n\
         <|im_start|>assistant\n",
        prompt
    );

    // 编码
    let prompt_tokens = tokenizer.encode(formatted_prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {e}"))?;
    let input_tokens = prompt_tokens.get_ids();

    // text generation
    let seed = rand::random_range(200_000_000..400_000_000);
    let text_generation = TextGeneration::new(temperature, 1.1, 64, seed, top_p, None, false);

    // 执行生成
    let result = text_generation.generate(input_tokens, model, tokenizer, &Device::Cpu)?;

    Ok(result)
}
struct TextGeneration {
    temperature: f64,

    // 重复抑制
    // 重复惩罚系数（类似HF的repetition_penalty）：
    // - 值>1.0时抑制重复（如1.2增加20%重复token的loss）
    // - 值<1.0时允许重复（如0.8降低20%惩罚）
    repeat_penalty: f32,

    // 回溯窗口长度（类似HF的repeat_last_n）：
    // - 检查最近N个token是否重复
    // - 典型值64（平衡性能与抑制效果）
    repeat_last_n: usize,

    seed: u64,
    top_p: Option<f64>,
    top_k: Option<usize>,
    split_prompt: bool,
}

impl TextGeneration {
    pub fn new(
        temperature: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        top_p: Option<f64>,
        top_k: Option<usize>,
        split_prompt: bool,
    ) -> Self {
        Self {
            temperature,
            repeat_penalty,
            repeat_last_n,
            seed,
            top_p,
            top_k,
            split_prompt,
        }
    }

    pub fn generate(
        &self,
        input_tokens: &[u32],
        model: &mut ModelWeights,
        tokenizer: &Tokenizer,
        device: &Device,
    ) -> anyhow::Result<String> {
        let temperature = self.temperature;
        let mut logits_processor = {
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (self.top_k, self.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(self.seed, sampling)
        };

        let mut next_token = if !self.split_prompt {
            let input = Tensor::new(input_tokens, &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0).unwrap();
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        } else {
            let mut next_token = 0;
            for (pos, token) in input_tokens.iter().enumerate() {
                let input = Tensor::new(&[*token], &device)?.unsqueeze(0)?;
                let logits = model.forward(&input, pos)?;
                let logits = logits.squeeze(0)?;
                next_token = logits_processor.sample(&logits)?
            }
            next_token
        };
        let mut all_tokens = vec![];
        all_tokens.push(next_token);

        let mut tos = TokenOutputStream::new(tokenizer.clone());
        if let Some(t) = tos.next_token(next_token)? {
            // print!("{t}");
            // std::io::stdout().flush()?;
        }
        let eos_token = tos
            .tokenizer()
            .get_vocab(true)
            .get("<|im_end|>")
            .unwrap()
            .clone();

        for index in 0..999 {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, input_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token)? {
                // print!("{t}");
                // std::io::stdout().flush()?;
            }
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
            // println!("{rest}");
        }
        // std::io::stdout().flush()?;

        let result_msg = tos.tokenizer().decode(&all_tokens, true).unwrap();
        Ok(result_msg)
    }
}

// ---------------- Metrics collector & helpers ----------------

fn spawn_metrics_collector(metrics: Arc<Metrics>) {
    tokio::spawn(async move {
        let mut sys = System::new_all();
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        let mut net = Networks::new();
        let mut disks = Disks::new();
        loop {
            interval.tick().await;

            // 刷新数据
            sys.refresh_cpu_usage();
            sys.refresh_memory();
            disks.refresh(false);
            net.refresh(false);

            // CPU 使用率（sys.global_cpu_usage() 返回 f32）
            let cpu_pct = sys.global_cpu_usage();

            // 内存使用率
            let total_mem = sys.total_memory() as f32;
            let used_mem = (sys.total_memory() - sys.available_memory()) as f32;
            let mem_pct = if total_mem > 0.0 {
                used_mem / total_mem * 100.0
            } else {
                0.0
            };

            // 磁盘使用率（汇总所有磁盘）
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

            // 网络流量（累计接收 + 发送），计算近 5 秒带宽（bytes/sec -> Mbps）
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
            // 归一化为 0..100 的百分比表示（以 1000 Mbps 为 100% 的参考值）
            let net_pct = ((mbps / 1000.0) * 100.0).min(100.0) as f32;

            // push 到环形缓冲
            push(&metrics.cpu, cpu_pct).await;
            push(&metrics.memory, mem_pct).await;
            push(&metrics.disk, disk_pct).await;
            push(&metrics.network, net_pct).await;
        }
    });
}

/// helper: 将新样本压入 VecDeque（保证容量）
async fn push(m: &Mutex<VecDeque<f32>>, val: f32) {
    let mut dq = m.lock().await;
    if dq.len() >= SAMPLE_CAPACITY {
        dq.pop_front();
    }
    // 保持两位小数
    dq.push_back((val * 100.0).round() / 100.0);
}
// ---------------- system_stats_handler ----------------

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
// ---------------- history_handler ----------------

async fn history_handler(State(state): State<AppState>) -> impl IntoResponse {
    // 查询最近 20 条（按 id 倒序，最后再反转成正序，保证显示从旧到新）
    let rows = sqlx::query("SELECT role, content FROM chat_history ORDER BY id DESC LIMIT 20")
        .fetch_all(&state.db)
        .await
        .unwrap_or_default();

    let history: Vec<Value> = rows
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
