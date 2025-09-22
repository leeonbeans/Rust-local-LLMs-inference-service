use actix_web::{web, App, HttpServer, HttpResponse, Result};
use actix_files::NamedFile;
use std::path::PathBuf;
use tera::{Tera, Context};
use serde_json;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::time::Duration;
use actix_web::middleware::Logger;

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
    success: bool,
}

// 首页处理器
async fn index(tera: web::Data<Tera>) -> Result<HttpResponse> {
    let mut context = Context::new();
    context.insert("title", "千问模型交互界面");

    let rendered = tera.render("index.html", &context)
        .map_err(|e| {
            eprintln!("Template error: {}", e);
            actix_web::error::ErrorInternalServerError("Template error")
        })?;

    Ok(HttpResponse::Ok().content_type("text/html").body(rendered))
}

// 聊天API处理器
async fn chat(
    web::Json(data): web::Json<ChatRequest>,
) -> Result<HttpResponse> {
    println!("收到消息: {}", data.message);
    println!("参数 - temperature: {:?}, max_tokens: {:?}", data.temperature, data.max_tokens);

    // 构建消息历史（这里简化处理，实际应该维护对话历史）
    let messages = vec![
        serde_json::json!({
            "role": "system",
            "content": "你是一个乐于助人的AI助手。请用中文回答用户的问题。"
        }),
        serde_json::json!({
            "role": "user",
            "content": data.message
        })
    ];

    // 尝试连接到llama.cpp
    let client = Client::new();
    let endpoints = [
        "http://localhost:8080/v1/chat/completions",
        "http://localhost:8080/completion",
    ];

    for endpoint in endpoints.iter() {
        println!("尝试连接到: {}", endpoint);

        // 根据不同端点构建不同的请求体
        let request_body = if endpoint.contains("/v1/chat/completions") {
            // 新的聊天补全格式
            serde_json::json!({
                "messages": messages,
                "temperature": data.temperature.unwrap_or(0.7),
                "max_tokens": data.max_tokens.unwrap_or(-1),
                "stop": ["\n", "User:", "用户:"]
            })
        } else {
            // 旧的补全格式（兼容性）
            serde_json::json!({
                "prompt": data.message,
                "temperature": data.temperature.unwrap_or(0.7),
                "max_tokens": data.max_tokens.unwrap_or(512),
                "stop": ["\n", "User:", "用户:"]
            })
        };

        match client
            .post(*endpoint)
            .json(&request_body)
            .timeout(Duration::from_secs(30))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                match response.json::<serde_json::Value>().await {
                    Ok(llama_response) => {
                        println!("成功从llama.cpp获取响应: {:?}", llama_response);

                        // 提取响应文本 - 根据不同的响应格式处理
                        let response_text = if endpoint.contains("/v1/chat/completions") {
                            // 新的聊天格式响应
                            llama_response["choices"][0]["message"]["content"]
                                .as_str()
                                .unwrap_or("")
                                .to_string()
                        } else if let Some(content) = llama_response["content"].as_str() {
                            // 旧的内容字段
                            content.to_string()
                        } else if let Some(text) = llama_response["text"].as_str() {
                            // 旧的文本字段
                            text.to_string()
                        } else if let Some(choices) = llama_response["choices"].as_array() {
                            // 其他可能的choices格式
                            choices.first()
                                .and_then(|choice| choice["text"].as_str())
                                .unwrap_or("")
                                .to_string()
                        } else {
                            format!("收到响应但无法解析格式: {}", llama_response)
                        };

                        return Ok(HttpResponse::Ok()
                            .json(ChatResponse {
                                response: response_text,
                                success: true,
                            }));
                    }
                    Err(e) => {
                        eprintln!("解析llama.cpp响应失败: {}", e);
                    }
                }
            }
            Ok(response) => {
                eprintln!("llama.cpp返回错误状态: {}", response.status());
                if let Ok(body) = response.text().await {
                    eprintln!("错误响应体: {}", body);
                }
            }
            Err(e) => {
                eprintln!("连接到llama.cpp失败: {}", e);
            }
        }
    }

    // 如果所有连接尝试都失败，返回演示响应
    let demo_response = if data.message.contains("你好") || data.message.contains("hello") {
        "您好！我是AI助手。目前无法连接到后端模型服务，请确保llama.cpp在8080端口运行。"
    } else {
        "我已收到您的消息。目前无法连接到AI模型服务，请检查llama.cpp是否在8080端口运行。"
    };

    Ok(HttpResponse::Ok()
        .json(ChatResponse {
            response: demo_response.to_string(),
            success: false,
        }))
}

// 检查连接状态
async fn check_connection() -> Result<HttpResponse> {
    let client = Client::new();
    let mut results = Vec::new();

    // 测试llama.cpp连接
    let llama_endpoints = [
        "http://localhost:8080",
        "http://localhost:8080/completion",
        "http://localhost:8080/v1/models",
        "http://localhost:8080/v1/chat/completions",
    ];

    for endpoint in llama_endpoints.iter() {
        match client.get(*endpoint).timeout(Duration::from_secs(3)).send().await {
            Ok(response) => {
                results.push(format!("{}: ✅ 成功 (状态: {})", endpoint, response.status()));
            }
            Err(e) => {
                results.push(format!("{}: ❌ 失败 ({})", endpoint, e));
            }
        }
    }

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "connection_test",
        "results": results,
        "server": "Rust服务器运行在8081端口",
        "timestamp": chrono::Local::now().to_rfc3339(),
    })))
}

// 健康检查端点
async fn health_check() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "service": "llama-ui-proxy",
        "port": 8081,
        "timestamp": chrono::Local::now().to_rfc3339(),
    })))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // 启用日志
    std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();

    println!("启动代理服务器: http://localhost:8081");
    println!("将尝试连接到llama.cpp服务: http://localhost:8080");
    println!("支持的消息格式:");
    println!("  - /v1/chat/completions (推荐)");
    println!("  - /completion (兼容模式)");
    println!("如果连接失败，将使用演示模式");

    HttpServer::new(|| {
        let tera = Tera::new("templates/**/*").unwrap_or_else(|e| {
            eprintln!("模板初始化失败: {}", e);
            std::process::exit(1);
        });

        App::new()
            .wrap(Logger::default())
            .wrap(Logger::new("%a %{User-Agent}i"))
            .app_data(web::Data::new(tera))
            .route("/", web::get().to(index))
            .route("/api/chat", web::post().to(chat))
            .route("/api/check-connection", web::get().to(check_connection))
            .route("/api/health", web::get().to(health_check))
            .service(actix_files::Files::new("/static", "static"))
    })
        .bind("127.0.0.1:8081")?
        .run()
        .await
}