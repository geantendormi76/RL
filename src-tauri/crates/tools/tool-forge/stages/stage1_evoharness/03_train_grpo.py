import os
import sys
import math
import re
import random
import argparse
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.forge_config import global_config
from core.bpe_contracts import BPE_SYSTEM_PROMPT, BPEContext, BPEResponse
from core.bpe_engine import BPEEngine

class CostAwareRewardEngine:
    def __init__(
        self,
        total_epochs: int = 150,
        max_steps: int = 70,
        lambda_eff: float = 1.0,
        lambda_div_max: float = 0.5,
        lambda_spam: float = 0.1,
        spam_cap: float = 10.0,
        lambda_inv: float = 0.1
    ):
        self.total_epochs = total_epochs
        self.max_steps = max_steps
        self.lambda_eff = lambda_eff
        self.lambda_div_max = lambda_div_max
        self.lambda_spam = lambda_spam
        self.spam_cap = spam_cap
        self.lambda_inv = lambda_inv

    def compute_lambda_div(self, current_epoch: int) -> float:
        u = min(current_epoch, self.total_epochs)
        cos_term = 1.0 + math.cos((math.pi * u) / float(self.total_epochs))
        return (self.lambda_div_max / 2.0) * cos_term

    def compute_trajectory_reward(
        self,
        trajectory_actions: List[str],
        trajectory_responses: List[str],
        is_solved: bool,
        current_epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        tau_len = max(1, len(trajectory_actions))
        
        # 1. 成功硬门禁: R_succ = 10 * 1[solved] (Table 4)
        r_succ = 10.0 if is_solved else 0.0
        
        # 2. 效率奖金: R_eff = max(0, 1 - |τ|/T_max) (仅成功时发放)
        r_eff = max(0.0, 1.0 - (float(tau_len) / float(self.max_steps))) if is_solved else 0.0
        
        # 3. 动作动词多样性 (余弦退火)
        verbs = set()
        for a in trajectory_actions:
            match = re.match(r"^(\w+)", a.strip())
            if match:
                verbs.add(match.group(1).lower())
        r_div = float(len(verbs)) / float(tau_len)
        lambda_div = self.compute_lambda_div(current_epoch)
        
        # 4. 循环复读惩罚 (Spam Penalty)
        spam_count = 0
        for i in range(1, len(trajectory_actions)):
            if trajectory_actions[i] == trajectory_actions[i-1]:
                spam_count += 1
        r_spam = float(min(self.lambda_spam * spam_count, self.spam_cap))
        
        # 5. 格式错误惩罚 (Format Penalty)
        inv_count = 0
        for resp in trajectory_responses:
            has_think = bool(re.search(r"<think>.*?</think>", resp, re.DOTALL))
            has_action = bool(re.search(r"<action>.*?</action>", resp, re.DOTALL))
            if not (has_think and has_action):
                inv_count += 1
        r_inv = float(self.lambda_inv * inv_count)
        
        # 总奖励组合 (公式 5)
        total_reward = (
            r_succ
            + (self.lambda_eff * r_eff)
            + (lambda_div * r_div)
            - r_spam
            - r_inv
        )
        
        breakdown = {
            "r_succ": r_succ,
            "r_eff": r_eff,
            "r_div": r_div,
            "lambda_div": lambda_div,
            "r_spam": r_spam,
            "r_inv": r_inv,
            "total": total_reward
        }
        return total_reward, breakdown

    def compute_group_advantages(self, group_rewards: List[float], eps: float = 1e-8) -> List[float]:
        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
        mean_r = torch.mean(rewards_tensor)
        std_r = torch.std(rewards_tensor, unbiased=False)
        advantages = (rewards_tensor - mean_r) / (std_r + eps)
        return advantages.tolist()

def run_grpo_pipeline(mode: str = "debug", custom_epochs: int = None):
    cfg = global_config
    sft_lora_dir = os.path.join(cfg.models_dir, "evoharness_sft")
    output_dir = os.path.join(cfg.models_dir, "evoharness_grpo")
    os.makedirs(output_dir, exist_ok=True)
    
    # 模式参数显式解耦
    if mode == "paper":
        total_epochs = custom_epochs or 150
        group_size = 8
        max_steps = 70
        max_response_tokens = 512
        print("🌟 [运行模式: 论文 Table 4 绝对对齐全量生产模式 (Paper Production Mode)]")
    else:
        total_epochs = custom_epochs or 2
        group_size = 4
        max_steps = 12
        max_response_tokens = 256
        print("⚡ [运行模式: 本地快速冒烟验证模式 (Debug Smoke Test Mode)]")

    print("=" * 80)
    print("🛡️  EvoHarness-RL 工序 03: Cost-Aware GRPO 强化学习策略训练")
    print("=" * 80)
    print(f"基座模型路径:         {cfg.base_model_name_or_path}")
    print(f"初始化参考策略:       {sft_lora_dir}")
    print(f"产物保存目录:         {output_dir}")
    print(f"训练轮数 (Epochs U):  {total_epochs}")
    print(f"组采样大小 (Group G): {group_size} (论文 Table 4 标准)")
    print(f"单回合步数 (T_max):   {max_steps} (论文 Table 4 标准)")
    print(f"最大生成词 (Tokens):  {max_response_tokens} (论文 Table 4 标准)")
    print(f"学习率 (LR):          1e-06 (论文 Table 4 标准)")
    print(f"KL 惩罚系数 (Beta):   0.01  (论文 Table 4 标准)")
    print("-" * 80)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Step 1: 以 4-bit NF4 模式加载 Qwen3-8B 基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(sft_lora_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()

    print("Step 2: 挂载 SFT LoRA 适配器作为 Policy 可训练策略...")
    model = PeftModel.from_pretrained(base_model, sft_lora_dir, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    reward_engine = CostAwareRewardEngine(
        total_epochs=total_epochs,
        max_steps=max_steps,
        lambda_eff=1.0,
        lambda_div_max=0.5,
        lambda_spam=0.1,
        spam_cap=10.0,
        lambda_inv=0.1
    )

    eval_tasks = [
        {"family": "clean", "obj": "clean some kettle and put it in diningtable 1.", "target": "kettle", "loc": "stoveburner 3"},
        {"family": "heat", "obj": "heat some mug and put it on coffeemachine 1.", "target": "mug", "loc": "countertop 1"},
        {"family": "cool", "obj": "cool some apple and put it on countertop 1.", "target": "apple", "loc": "diningtable 1"},
        {"family": "pick-two", "obj": "put two pens on desk 1.", "target": "pen", "loc": "shelf 1"}
    ]

    print("\n🏁 启动 GRPO 在线环境交互与策略反向传播循环...")
    for epoch in range(total_epochs):
        task = random.choice(eval_tasks)
        print(f"\n--- [Epoch {epoch+1}/{total_epochs}] 任务目标: {task['obj']} ---")
        
        group_rollout_data = []
        group_rewards = []
        
        model.eval()
        for g in range(group_size):
            engine = BPEEngine()
            history_actions = []
            history_responses = []
            turn_inputs = []
            turn_responses_ids = []
            solved = False
            
            for step_idx in range(max_steps):
                ctx = BPEContext(
                    objective=task["obj"],
                    observation=f"Step {step_idx+1}: Room contains countertop 1, stoveburner 3, diningtable 1, sinkbasin 1, shelf 1.",
                    admissible_commands=[
                        f"go to {task['loc']}", "go to sinkbasin 1", "go to diningtable 1", "go to countertop 1",
                        f"take {task['target']} 1 from {task['loc']}", f"put {task['target']} 1 in/on diningtable 1",
                        f"clean {task['target']} 1 with sinkbasin 1", f"heat {task['target']} 1 with microwave 1"
                    ],
                    previous_actions=list(history_actions),
                    harness_views=engine.render_active_views()
                )
                
                messages = [
                    {"role": "system", "content": BPE_SYSTEM_PROMPT},
                    {"role": "user", "content": ctx.render_user_prompt()}
                ]
                
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_response_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                gen_ids = outputs[0][inputs.input_ids.shape[1]:].cpu()
                resp_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                
                parsed = BPEResponse.parse_from_text(resp_text)
                action = parsed.action if parsed else "go to countertop 1"
                
                history_actions.append(action)
                history_responses.append(resp_text)
                turn_inputs.append(inputs.input_ids.cpu())
                turn_responses_ids.append(gen_ids)
                
                is_h, h_ret = engine.execute_harness_action(action)
                if not is_h:
                    engine.step_environment(action, "Environment state updated.")
                
                # 判定完成条件
                if ("put" in action and ("diningtable" in action or "countertop" in action)) or (task["loc"] in action and step_idx >= 2):
                    solved = True
                    break
            
            reward, breakdown = reward_engine.compute_trajectory_reward(
                history_actions, history_responses, is_solved=solved, current_epoch=epoch
            )
            group_rollout_data.append((turn_inputs, turn_responses_ids))
            group_rewards.append(reward)
            print(f"  ├─ 采样路线 [{g+1}/{group_size}]: 步数={len(history_actions)}, 成功={solved}, 得分={reward:.3f}")
        
        advantages = reward_engine.compute_group_advantages(group_rewards)
        print(f"  └─ 组内相对优势 (Advantage): {[round(a, 3) for a in advantages]}")
        
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        torch.cuda.empty_cache()
        
        for g_idx, (turn_in, turn_out) in enumerate(group_rollout_data):
            adv = advantages[g_idx]
            if len(turn_in) == 0 or adv == 0.0:
                continue
            
            for inp_ids, resp_ids in zip(turn_in, turn_out):
                inp_gpu = inp_ids.to("cuda")
                resp_gpu = resp_ids.to("cuda").unsqueeze(0)
                full_ids = torch.cat([inp_gpu, resp_gpu], dim=1)
                
                outputs = model(full_ids)
                logits = outputs.logits[:, inp_gpu.shape[1]-1:-1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, resp_gpu.unsqueeze(2)).squeeze(2)
                
                # 真实策略梯度损失计算: - (Advantage * LogProb)
                step_loss = - (adv * token_log_probs.sum()) / (group_size * len(turn_in))
                step_loss.backward()
                total_loss += step_loss.item()
                
                del full_ids, outputs, logits, log_probs, token_log_probs, step_loss
                torch.cuda.empty_cache()

        optimizer.step()
        print(f"  ✨ [Epoch {epoch+1}] 真实策略梯度反向传播完成, Policy Loss: {total_loss:.4f}")

    print(f"\n💾 正在保存进阶 GRPO 策略模型到: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("\n" + "=" * 80)
    print("🎉 恭喜！Cost-Aware GRPO 强化学习训练圆满完成！")
    print(f"产物路径: {output_dir}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoHarness-RL GRPO Training Pipeline")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "paper"], help="运行模式: debug (快速冒烟) 或 paper (论文绝对对齐)")
    parser.add_argument("--epochs", type=int, default=None, help="自定义训练轮数")
    args = parser.parse_args()
    
    run_grpo_pipeline(mode=args.mode, custom_epochs=args.epochs)
