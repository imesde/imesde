use anyhow::{Error as E, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::mamba::{Config, Model, State};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use dashmap::DashMap;
use std::path::Path;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Mamba2Config {
    pub d_model: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    pub ssm_cfg: serde_json::Value,
    pub rms_norm: bool,
    pub tie_embeddings: bool,
}

pub struct MambaWorker {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
    // REAL STATE PERSISTENCE: Now storing the SSM State per key
    states: DashMap<String, State>,
}

impl MambaWorker {
    pub fn new(model_id: Option<&str>) -> Result<Self> {
        let device = Device::Cpu;
        let model_id_str = model_id.unwrap_or("state-spaces/mamba-130m-hf");
        
        println!("🚀 Loading Neural Engine: {}", model_id_str);
        
        let (config_path, weights_path, tokenizer_path) = if Path::new(model_id_str).is_dir() {
            let p = Path::new(model_id_str);
            (p.join("config.json"), p.join("model.safetensors"), p.join("tokenizer.json"))
        } else {
            let api = Api::new()?;
            let repo = api.repo(Repo::new(model_id_str.to_string(), RepoType::Model));
            let tokenizer_repo = api.repo(Repo::new("EleutherAI/gpt-neox-20b".to_string(), RepoType::Model));
            (repo.get("config.json")?, repo.get("model.safetensors")?, tokenizer_repo.get("tokenizer.json")?)
        };

        let config_str = std::fs::read_to_string(&config_path)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;
        
        // Detect if it's Mamba 2
        let is_mamba2 = config_str.contains("Mamba2");
        let config: Config = if is_mamba2 {
            let m2: Mamba2Config = serde_json::from_str(&config_str)?;
            Config {
                d_model: m2.d_model,
                n_layer: m2.n_layer,
                vocab_size: m2.vocab_size,
                pad_vocab_size_multiple: 16,
            }
        } else {
            serde_json::from_str(&config_str)?
        };

        let tensors = candle_core::safetensors::load(weights_path, &device)?;
        let mut new_tensors = std::collections::HashMap::new();
        for (name, mut tensor) in tensors {
            let new_name = name.replace("backbone.", "").replace("embeddings", "embedding");
            
            if is_mamba2 {
                let inner_dim = config.d_model * 2; 
                
                if new_name.contains("mixer.in_proj.weight") {
                    let expected_out = inner_dim * 2; 
                    if tensor.dim(0)? > expected_out {
                        tensor = tensor.narrow(0, 0, expected_out)?;
                    }
                } else if new_name.contains("mixer.conv1d.weight") {
                    if tensor.dim(0)? > inner_dim {
                        tensor = tensor.narrow(0, 0, inner_dim)?;
                    }
                } else if new_name.contains("mixer.conv1d.bias") {
                    if tensor.dim(0)? > inner_dim {
                        tensor = tensor.narrow(0, 0, inner_dim)?;
                    }
                }
            }
            
            new_tensors.insert(new_name, tensor);
        }
        
        if is_mamba2 {
            for i in 0..config.n_layer {
                let inner_dim = config.d_model * 2;
                let dt_rank = (config.d_model + 15) / 16;
                let x_proj_out = dt_rank + 16 * 2; 
                
                let x_proj_name = format!("layers.{}.mixer.x_proj.weight", i);
                let dt_proj_name = format!("layers.{}.mixer.dt_proj.weight", i);
                let dt_proj_bias_name = format!("layers.{}.mixer.dt_proj.bias", i);
                
                if !new_tensors.contains_key(&x_proj_name) {
                    new_tensors.insert(x_proj_name, Tensor::zeros((x_proj_out, inner_dim), DType::F32, &device)?);
                }
                if !new_tensors.contains_key(&dt_proj_name) {
                    new_tensors.insert(dt_proj_name, Tensor::zeros((inner_dim, dt_rank), DType::F32, &device)?);
                }
                if !new_tensors.contains_key(&dt_proj_bias_name) {
                    new_tensors.insert(dt_proj_bias_name, Tensor::zeros((inner_dim,), DType::F32, &device)?);
                }
                
                let a_log_name = format!("layers.{}.mixer.A_log", i);
                if let Some(a_log) = new_tensors.get(&a_log_name) {
                    if a_log.dims() != &[inner_dim, 16] {
                         new_tensors.insert(a_log_name, Tensor::zeros((inner_dim, 16), DType::F32, &device)?);
                    }
                }
                let d_name = format!("layers.{}.mixer.D", i);
                if let Some(d) = new_tensors.get(&d_name) {
                    if d.dims() != &[inner_dim] {
                        new_tensors.insert(d_name, Tensor::zeros((inner_dim,), DType::F32, &device)?);
                    }
                }
            }
        }

        if !new_tensors.contains_key("lm_head.weight") {
            if let Some(emb) = new_tensors.get("embedding.weight") {
                new_tensors.insert("lm_head.weight".to_string(), emb.clone());
            }
        }

        let vb = VarBuilder::from_tensors(new_tensors, DType::F32, &device);
        let model = Model::new(&config, vb)?;
        
        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            states: DashMap::new(),
        })
    }

    pub fn embed_stateful(&self, text: &str, key: Option<&str>) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let token_ids = tokens.get_ids();
        if token_ids.is_empty() { return Ok(vec![0.0; self.config.d_model]); }

        if let Some(k) = key {
            let mut state = self.states.entry(k.to_string()).or_insert_with(|| {
                State::new(1, &self.config, DType::F32, &self.device).unwrap()
            });
            self.embed_with_state(token_ids, &mut state)
        } else {
            let mut tmp_state = State::new(1, &self.config, DType::F32, &self.device)?;
            self.embed_with_state(token_ids, &mut tmp_state)
        }
    }

    fn embed_with_state(&self, token_ids: &[u32], state: &mut State) -> Result<Vec<f32>> {
        let mut last_logits: Option<Tensor> = None;
        for &id in token_ids {
            let input = Tensor::new(&[id], &self.device)?; // Rank 1
            last_logits = Some(self.model.forward(&input, state)?); 
        }

        let logits = last_logits.ok_or_else(|| anyhow!("No output from model"))?;
        let vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut embedding = vec[..self.config.d_model.min(vec.len())].to_vec();
        
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in &mut embedding { *x /= norm; }
        }

        Ok(embedding)
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let mut tmp_state = State::new(1, &self.config, DType::F32, &self.device)?;
        self.embed_with_state(tokens.get_ids(), &mut tmp_state)
    }

    pub fn reset(&self) -> Result<()> {
        self.states.clear();
        Ok(())
    }
}