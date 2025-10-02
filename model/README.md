Here's GGUF model folder.
If you want to use model with llama.cpp, please do the steps as follows.
1. Download models, whose format should be GGUF (like foo.gguf).
2. Put them in the model folder.
3. In `/src/main.rs`, below `let mut model_paths = HashMap::new();` add code `model_paths.insert("MODEL_NAME".to_string(), "MODEL_PATH".to_string());`.

For example,
```tips
model_paths.insert("foo".to_string(), ".\\model\\foo.gguf".to_string());
```

4.In `/static/index.html`, under `<select id="model-select">` add code `<option value="MODEL_NAME">MODEL_NAME</option>`.

For example,
```tips
<option value="foo">foo</option>
```
ATTENTION, the option value needs to be the same as the model_name in `model_paths`.
