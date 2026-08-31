import os
import sys
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.forge_config import global_config
from core.bpe_contracts import BPE_SYSTEM_PROMPT, BPEContext

def verify_sft_generation():
    cfg = global_config
    lora_dir = os.path.join(cfg.models_dir, "evoharness_sft")
    
    print("=" * 80)
    print("🔬 SFT 微调模型实机推理探针验证 (Inference Probe - 512 Tokens)")
    print("=" * 80)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("正在加载 Qwen3-8B + SFT LoRA 适配器...")
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    test_ctx = BPEContext(
        objective="clean some pan and put it on diningtable 1.",
        observation="You are in a modern kitchen. In front of you is stoveburner 1, sinkbasin 1, countertop 1, diningtable 1.",
        admissible_commands=["go to stoveburner 1", "go to sinkbasin 1", "go to countertop 1", "go to diningtable 1"],
        previous_actions=[],
        harness_views="# PLAN: []\n# RECALLED HINTS: None"
    )

    messages = [
        {"role": "system", "content": BPE_SYSTEM_PROMPT},
        {"role": "user", "content": test_ctx.render_user_prompt()}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    print("\n🚀 正在进行贪婪解码生成 (Greedy Decoding, max_new_tokens=512)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    print("\n" + "=" * 80)
    print("【SFT 微调模型实机生成完整结果】:")
    print(response_text)
    print("=" * 80)

    has_think = bool(re.search(r"<think>.*?</think>", response_text, re.DOTALL))
    has_action = bool(re.search(r"<action>.*?</action>", response_text, re.DOTALL))
    is_valid_format = has_think and has_action

    print(f"1. 是否包含完整闭合 <think> 思维链: {has_think}")
    print(f"2. 是否包含完整闭合 <action> 动作标签: {has_action}")
    print(f"3. 整体格式符合度: {'100% 合规' if is_valid_format else '不合规'}")
    print("-" * 80)

    assert is_valid_format, "生成结果必须严格包含闭合的 <think> 和 <action> 标签!"
    print("✅ 探针断言通过：SFT 模型已完全掌握 BPE 认知动作与思考格式！\n")

if __name__ == "__main__":
    verify_sft_generation()
