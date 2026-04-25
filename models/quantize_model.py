import os
import torch
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def quantize_and_save():
    model_dir = 'models/final_optimized_model/final_model'
    tokenizer_dir = 'models/final_optimized_model/final_tokenizer'
    output_dir = 'models/quantized_model'
    
    print(f"Loading unquantized model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Original Size
    orig_size = 0
    for param in model.parameters():
        orig_size += param.nelement() * param.element_size()
    print(f"Original model memory size: {orig_size / 1024**2:.2f} MB")
    
    # Apply dynamic quantization to Linear layers
    print("Applying INT8 dynamic quantization to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # New Size (estimation based on state dict serialization)
    print("Saving quantized model...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Actually checking disk size of the pytorch_model.bin/model.safetensors
    orig_file_size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)))
    quant_file_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)))
    
    print(f"\\nOptimization Complete!")
    print(f"- Original Disk Size: {orig_file_size / 1024**2:.2f} MB")
    print(f"- Quantized Disk Size: {quant_file_size / 1024**2:.2f} MB")
    print(f"- Saved to: {output_dir}")

if __name__ == "__main__":
    quantize_and_save()
