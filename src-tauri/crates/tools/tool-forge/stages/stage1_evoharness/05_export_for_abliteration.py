import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.forge_config import global_config

def export_merged_bf16_model():
    cfg = global_config
    
    # 优先融合 GRPO 进阶适配器，若无则退回融合 SFT 适配器
    grpo_dir = os.path.join(cfg.models_dir, "evoharness_grpo")
    sft_dir = os.path.join(cfg.models_dir, "evoharness_sft")
    
    if os.path.exists(os.path.join(grpo_dir, "adapter_model.safetensors")):
        lora_dir = grpo_dir
        print("🌟 [检测到 GRPO 强化学习进阶适配器，采用最高智商模型进行融合！]")
    else:
        lora_dir = sft_dir
        print("⚡ [检测到 SFT 基础适配器，进行融合...]")

    export_dir = os.path.join(cfg.models_dir, "handoff_for_abliteration")
    
    print("=" * 80)
    print("🛡️  EvoHarness-RL 工序 05: LoRA-to-BF16 无损模型融合与消融工厂交付")
    print("=" * 80)
    print(f"原始纯血基座:   {cfg.base_model_name_or_path}")
    print(f"LoRA 适配器路径: {lora_dir}")
    print(f"消融交付路径:   {export_dir}")
    print("-" * 80)

    print("Step 1: 正在以纯血 BF16 精度载入原始未量化的 Qwen3-8B 基座 (CPU 内存安全加载)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, trust_remote_code=True)

    print("Step 2: 挂载进阶 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, lora_dir)

    print("Step 3: 执行 W_merged = W_original_bf16 + (alpha/r)*(B·A) 1:1 精确数学融合...")
    merged_model = model.merge_and_unload()

    print(f"Step 4: 正在以标准 Safetensors 格式保存纯血 BF16 模型至: {export_dir}")
    os.makedirs(export_dir, exist_ok=True)
    merged_model.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)

    print("\n" + "=" * 80)
    print("🎉 恭喜！阶段一全生命周期（SFT 语法对齐 + GRPO 策略进化 + LoRA-to-BF16 无损总装）全部圆满完成！")
    print(f"交付物位置: {export_dir}")
    print("该模型具备顶尖 Agent 决策脑力且底座 0 量化失真，可直接无缝交付 LLM_pox 消融工厂！")
    print("=" * 80 + "\n")
    return True

if __name__ == "__main__":
    export_merged_bf16_model()
