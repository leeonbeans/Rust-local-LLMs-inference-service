use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
// 导入 Model 类型
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::{io::Write, path::PathBuf, str::FromStr};
use tokenizers::Tokenizer;
//千问量化模型
use candle_transformers::models::quantized_qwen2::ModelWeights;
use rand::prelude::*;
use anyhow::Result;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("is_cuda={}", device.is_cuda());

    let (mut model, tokenizer) = get_model_and_tokenizer(
        &device,
        "F:/shixun/model-fast/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "F:/shixun/model-fast/tokenizer.json",
    )
        .unwrap();
    println!("model and tokenizer load ok");



    // The seed to use when generating random samples.
    // 不同的seed会导致不同的输出
    let seed = rand::random_range(200_000_000..400_000_000);

    let text_generation = TextGeneration::new(0.8, 1.1, 64, seed, None, None, false);


    let token_ids =
        get_prompt_tokens("我有一个苹果,一个橙子,一颗白菜 请给它们分类", &tokenizer).unwrap();
    println!(
        "{}",
        text_generation
            .generate(&token_ids, &mut model, &tokenizer, &device)
            .unwrap()
    );

    println!("==========================================================");



    let token_ids = get_prompt_tokens("'洋娃娃和小熊跳舞' 下一句是什么?", &tokenizer).unwrap();
    println!(
        "{}",
        text_generation
            .generate(&token_ids, &mut model, &tokenizer, &device)
            .unwrap()
    );


    println!("==========================================================");

    let prompt_str =
        get_prompt_str("帮我发邮件给Tom, 内容是:有内鬼,交易终止! Tom的邮箱地址是: tom@example.com");
    let token_ids = get_prompt_tokens(&prompt_str, &tokenizer).unwrap();
    println!(
        "{}",
        text_generation
            .generate(&token_ids, &mut model, &tokenizer, &device)
            .unwrap()
    );


    println!("==========================================================");

    let prompt_str =
        get_prompt_str("帮我发短信给Tom, 内容是:有内鬼,交易终止! Tom的手机号是: 139000002222");
    let token_ids = get_prompt_tokens(&prompt_str, &tokenizer).unwrap();
    println!(
        "{}",
        text_generation
            .generate(&token_ids, &mut model, &tokenizer, &device)
            .unwrap()
    );

    println!("==========================================================");

    Ok(())
}

fn get_prompt_str(user_input: &str) -> String {
    format!(
        r#"
```
{user_input}
```
你是一个AI助手, 可以帮我识别用户的意图, 请帮我识别上面的用户意图, 并帮我把关键信息整理下方样式的json格式:
```
{{
    "action": "{{意图枚举}}",
    "data": "{{关键信息}}"
}}
```
意图枚举: send_email, tel_phone
如果意图是send_email, 关键信息: {{ email: "xxx@xxxx", to_user: "xxx", message: "" }}
如果意图是tel_phone, 关键信息: {{ phone: "139xxxxxx", message: "" }}
    "#
    )
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

fn get_prompt_tokens(prompt_str: &str, tokenizer: &Tokenizer) -> anyhow::Result<Vec<u32>> {
    let prompt_str = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt_str
    );
    print!("formatted instruct prompt: {}", &prompt_str);

    let prompt_tokens = tokenizer.encode(prompt_str, true).unwrap();

    Ok(prompt_tokens.get_ids().to_vec())
}

fn get_model_and_tokenizer(
    device: &Device,
    model_path: &str,
    tokenizer_path: &str,
) -> anyhow::Result<(ModelWeights, Tokenizer)> {
    // 读取 GGUF 模型文件
    let model_path = PathBuf::from_str(model_path).unwrap();
    let mut model_file = std::fs::File::open(&model_path)?;

    let model = {
        let model =
            gguf_file::Content::read(&mut model_file).map_err(|e| e.with_path(model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({}) ",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
        );
        ModelWeights::from_gguf(model, &mut model_file, &device)?
    };

    // 创建 Tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    Ok((model, tokenizer))
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}