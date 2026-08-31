import os
import argparse
from core.forge_config import global_config

class PipelineOrchestrator:
    def __init__(self):
        self.config = global_config
        self.steps = [
            ("01_data", "SFT 专家轨迹与 BPE 样本构建", self.check_data),
            ("02_sft", "SFT 认知动作语义微调", self.check_sft),
            ("03_grpo", "Cost-aware GRPO 强化学习策略训练", self.check_grpo),
            ("04_eval", "BPE Agent 成功率评估与退火分析", self.check_eval),
            ("05_export", "模型权重打包交付 (供消融工厂使用)", self.check_export),
        ]

    def check_data(self) -> bool:
        return os.path.exists(os.path.join(self.config.data_dir, "bpe_sft_dataset.jsonl"))

    def check_sft(self) -> bool:
        sft_dir = os.path.join(self.config.models_dir, "evoharness_sft")
        return os.path.exists(os.path.join(sft_dir, "adapter_model.safetensors")) or os.path.exists(os.path.join(sft_dir, "adapter_config.json"))

    def check_grpo(self) -> bool:
        grpo_dir = os.path.join(self.config.models_dir, "evoharness_grpo")
        return os.path.exists(os.path.join(grpo_dir, "adapter_model.safetensors"))

    def check_eval(self) -> bool:
        return os.path.exists(os.path.join(self.config.outputs_dir, "eval_results.json"))

    def check_export(self) -> bool:
        export_dir = os.path.join(self.config.models_dir, "handoff_for_abliteration")
        return os.path.exists(export_dir) and len(os.listdir(export_dir)) > 0

    def print_status(self):
        print("\n" + "="*80)
        print("🛡️  EvoHarness-RL 认知训练与 BPE 运行时中枢看板 (Status Dashboard)")
        print("="*80)
        print(f"项目物理根目录: {self.config.project_root}")
        print(f"基座训练模型:   {self.config.base_model_name_or_path}")
        print(f"训练数据路径:   {self.config.data_dir}")
        print(f"模型产物路径:   {self.config.models_dir}")
        print("-" * 80)
        
        for step_id, step_desc, check_fn in self.steps:
            status_icon = "🟢 已完成" if check_fn() else "⚪ 未执行"
            print(f"  ├─ [{step_id}] {step_desc:<45} -> {status_icon}")
            
        print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoHarness-RL Pipeline Orchestrator")
    parser.add_argument("--status", action="store_true", help="查看流水线工序状态看板")
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator()
    orchestrator.print_status()
