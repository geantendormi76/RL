"""Stage 1: Open-ended Summarization Probe aligned with Hugging Face Space."""

import os
import sys
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.ui import ui, console
from rich.panel import Panel

def run_probe():
    model_dir = r"C:\dev\RL\models\Qwen3-VL-8B-Instruct"
    real_img_path = r"C:\dev\RL\data\service-ocr.png"

    ui.render_phase("01", "Qwen3-VL-8B 开放式核心总结 (对齐 HF 官方在线 Demo 行为)")
    console.print(f"  • 测试图片: [bold cyan]{real_img_path}[/]")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    console.print("  • 正在加载 Qwen3-VL-8B 处理器与 4-bit 权重...")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model.eval()

    # 1:1 对齐 Hugging Face 网页端的完全一致提问: "总结这张图片的核心内容"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": real_img_path},
                {"type": "text", "text": "总结这张图片的核心内容"}
            ]
        }
    ]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    console.print("[bold yellow]🚀 正在进行 GPU 视觉推理 (开放生成 max_new_tokens=1024)...[/]")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False  # 保持确定性解码，防止出现网页版“宝五金盛”颠倒错误
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    console.print(Panel(output_text, title="[bold bright_green]🎯 本地对齐 HF 提问后的输出结果[/]", border_style="bright_green"))

if __name__ == "__main__":
    run_probe()
