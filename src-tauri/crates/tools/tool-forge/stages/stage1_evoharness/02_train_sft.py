import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.forge_config import global_config

def run_sft():
    cfg = global_config
    sft_cfg = cfg.evoharness
    
    dataset_path = os.path.join(cfg.data_dir, "bpe_sft_dataset.jsonl")
    output_dir = os.path.join(cfg.models_dir, "evoharness_sft")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🛡️  EvoHarness-RL 工序 02: SFT 认知动作语义微调")
    print("=" * 80)
    print(f"基座模型:       {cfg.base_model_name_or_path}")
    print(f"训练数据路径:   {dataset_path}")
    print(f"模型产物路径:   {output_dir}")
    print(f"LoRA 参数:      r={sft_cfg.lora_r}, alpha={sft_cfg.lora_alpha}")
    print(f"批次大小与累积: batch_size={sft_cfg.sft_batch_size}, accum={sft_cfg.sft_grad_accum_steps}")
    print("-" * 80)

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"加载数据集成功，共 {len(dataset)} 条样本。")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("正在以 4-bit NF4 模式加载 Qwen3-8B 权重至 GPU (显存保护)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=sft_cfg.lora_r,
        lora_alpha=sft_cfg.lora_alpha,
        lora_dropout=sft_cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg.sft_epochs,
        per_device_train_batch_size=sft_cfg.sft_batch_size,
        gradient_accumulation_steps=sft_cfg.sft_grad_accum_steps,
        learning_rate=sft_cfg.sft_learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        logging_steps=1,
        bf16=True,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        max_length=sft_cfg.sft_max_seq_len,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config
    )

    print("\n🚀 启动 SFT 训练循环...")
    trainer.train()

    print(f"\n💾 正在保存 SFT 适配器与 Tokenizer 到: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("=== 工序 02 SFT 微调成功完成! ===")

if __name__ == "__main__":
    run_sft()
