import torch
from safetensors.torch import save_file
import os

pth_path = "model/mamba2-130m/pytorch_model.bin"
st_path = "model/mamba2-130m/model.safetensors"

if os.path.exists(pth_path):
    print(f"🔄 Converting {pth_path} to Safetensors (handling shared tensors)...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    
    # Rimuoviamo i duplicati che condividono memoria per far felice safetensors
    # Rust gestirà il weight tying se manca lm_head.weight
    if 'lm_head.weight' in state_dict:
        del state_dict['lm_head.weight']
        print("ℹ️ Removed shared lm_head.weight (will be tied in Rust)")
        
    save_file(state_dict, st_path)
    print(f"✅ Created {st_path}")
else:
    print(f"❌ File {pth_path} not found.")