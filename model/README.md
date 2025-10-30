这里是GGUF模型文件夹。
如果你想要使用自定义的模型，请跟着下列步骤进行操作：

1. 下载GGUF格式的模型，即后缀名为.gguf。如果想使用candle驱动，请下载对应的tokenizer.json文件。（例如下载 千问3gguf 以及 千问3非gguf(形如model-00001-of-00002.safetensors) 文件夹下的 tokenizer.json。

2. 将他放置在`./model`文件夹下的任意位置。

3. 打开`/src/main.rs`。
   * 若要添加使用llama.cpp加载的模型，请在代码中搜寻`model_paths`变量，添加模组键值以及对应路径。`model_paths.insert("随意起的模型键值名".to_string(), "gguf文件路径位置".to_string());`
     
     例如：
     ```tips
     model_paths.insert("qwen3".to_string(), ".\\model\\qwen3-0.5b.gguf".to_string());
     ```
   * 若要添加使用candle加载的模型，请在代码中搜寻`candle_models`变量，添加模组键值以及对应路径。`candle_models.insert("随意起的模型键值名".to_string(),("gguf模型路径".to_string(),"对应tokenizer.json文件路径".to_string(),),);`
     
     例如：
     ```tips
     candle_models.insert(
        "qwen3-candle".to_string(),
        (
            ".\\model\\candle\\qwen3\\qwen3.gguf".to_string(),
            ".\\model\\candle\\qwen3\\tokenizer.json".to_string(),
        ),
      );
     ```

4.打开`/static/index.html`，找到`<select id="model-select"></select>`，在中间添加对应的选项。`<option value="你上面起的模型键值名">想要在前端显示的名字</option>`

例如：
  ```tips
     <option value="qwen3">qwen3</option>
     <option value="qwen3-candle">qwen3-candle</option>
  ```

Here's GGUF model folder.
If you want to use model you downloaded , please do the steps as follows.
1. Download models, whose format should be GGUF (like foo.gguf). If you want to use candle, please another download its `tokenizer.json`.
   (PS. you can search first its non-gguf version, like `model-00001-of-00002.safetensors`. And you can find tokenizer.json in the folder. )
   
2. Put them in the model folder in your way.

3. In `/src/main.rs`:

    * If you want to use model with llama.cpp -->
      below `let mut model_paths = HashMap::new();` add code `model_paths.insert("MODEL_NAME".to_string(), "MODEL_PATH".to_string());`.

      For example,
      ```tips
        model_paths.insert("foo".to_string(), ".\\model\\foo.gguf".to_string());
      ```

    * If you want to use model with candle -->
      below `let mut candle_models = HashMap::new();` add code `candle_models.insert("MODEL_NAME".to_string(),("MODEL_PATH".to_string(),"MODEL_TOKENIZER_PATH".to_string(),),);`.

      For example,
      ```tips
        candle_models.insert(
        "foo-candle".to_string(),
        (
            ".\\model\\candle\\foo\\foo.gguf".to_string(),
            ".\\model\\candle\\foo\\tokenizer.json".to_string(),
        ),
      );
      ```

4.In `/static/index.html`, under `<select id="model-select">` add code `<option value="MODEL_NAME">MODEL_NAME</option>`.

For example,
```tips
<option value="foo">foo with llama.cpp</option>
<option value="foo-candle">foo with candle</option>
```
ATTENTION, the option value needs to be the same as the model_name in `model_paths` or `candle_models`.
