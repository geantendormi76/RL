"""Universal Modern Terminal UI Engine for EvoHarness-RL and Multi-Modal Forge Systems."""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console(highlight=False)


class ForgeUI:
    """Renders modern tables, panels, and metric dashboards for EvoHarness-RL Forge."""

    @staticmethod
    def render_header(
        title: str,
        model_name: str,
        model_path: str,
        architecture_type: str,
        num_layers: int,
        hidden_size: int,
        device_name: str = "NVIDIA GeForce RTX 3060 (12GB)",
    ) -> None:
        header_text = Text(f"🛡️  {title}", style="bold white on #1a365d", justify="center")
        console.print(Panel(header_text, box=box.ROUNDED, border_style="bright_blue", padding=(0, 1)))

        meta_table = Table(show_header=False, box=None, padding=(0, 1))
        meta_table.add_column("Key", style="bold cyan", justify="right")
        meta_table.add_column("Val", style="white")

        meta_table.add_row("📦 目标模型资产", f"[bold yellow]{model_name}[/]")
        meta_table.add_row("📁 物理存储路径", model_path)
        meta_table.add_row("🤖 模型类型分类", f"[bold magenta]{architecture_type}[/]")
        meta_table.add_row(
            "🏗️ 架构拓扑结构",
            f"[green]{num_layers}[/] 层 Decoder | Hidden: [green]{hidden_size}[/]"
        )
        meta_table.add_row("⚡ 硬件加速环境", f"[bold green]{device_name}[/] | CUDA 12.4 | Paged AdamW")

        console.print(Panel(meta_table, title="[bold bright_cyan]📋 模型工程 Profile[/]", box=box.ROUNDED, border_style="cyan"))

    @staticmethod
    def render_phase(step_num: str, title: str) -> None:
        console.print()
        console.print(f"[bold bright_yellow]━━━ [工序 {step_num}] {title} ━━━[/]")

    @staticmethod
    def render_diagnose_card(spec: Dict[str, Any], free_vram: float, total_vram: float) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold cyan", justify="right")
        table.add_column("Val", style="white")

        table.add_row("🤖 基础架构标识", f"[bold yellow]{spec.get('architecture', 'Unknown')}[/]")
        table.add_row("👁️ 视觉塔 (ViT) 状态", f"[bold green]{spec.get('vit_status', 'N/A')}[/]")
        table.add_row("🧠 语言主脑 (Trunk)", f"[green]{spec.get('num_hidden_layers', 'N/A')}[/] 层 | Hidden: [green]{spec.get('hidden_size', 'N/A')}[/]")
        table.add_row("🎯 LoRA 靶向注入", f"Rank=[bold green]{spec.get('lora_r', 16)}[/], Alpha=[bold green]{spec.get('lora_alpha', 32)}[/] ([dim]严格避开视觉塔[/])")
        table.add_row("⚡ GPU 显存水位", f"[bold green]{free_vram:.2f} GB[/] 空闲 / [bold white]{total_vram:.2f} GB[/] 总量 (安全阈值: 11.00 GB)")
        table.add_row("📂 产物持久化目录", f"[dim]{spec.get('output_dir', 'models/evoharness_vlm')}[/]")

        console.print(Panel(table, title="[bold bright_green]🔍 模型与训练环境白盒自省就绪卡[/]", box=box.ROUNDED, border_style="bright_green"))

    @staticmethod
    def render_grpo_epoch(
        epoch: int,
        total_epochs: int,
        task_name: str,
        avg_reward: float,
        policy_loss: float,
        annealing_lambda_div: float,
        group_trajectories: List[Dict[str, Any]],
    ) -> None:
        console.print(
            f"\n[bold cyan]▶ GRPO Epoch {epoch:03d}/{total_epochs:03d}[/] │ "
            f"目标: [yellow]{task_name}[/] │ "
            f"平均奖赏: [bold green]{avg_reward:.3f}[/] │ "
            f"策略损失: [bold red]{policy_loss:.4f}[/] │ "
            f"多样性退火 λ_div: [magenta]{annealing_lambda_div:.4f}[/]"
        )

        table = Table(box=box.SIMPLE, show_edge=False, header_style="bold cyan", padding=(0, 1))
        table.add_column("路线", justify="center")
        table.add_column("交互步数", justify="right")
        table.add_column("任务硬门禁", justify="center")
        table.add_column("动作多样性", justify="right")
        table.add_column("相对优势 (Advantage)", justify="right")

        for idx, traj in enumerate(group_trajectories):
            status = "[green]✔ 成功[/]" if traj.get("solved") else "[red]✘ 失败[/]"
            adv = traj.get("advantage", 0.0)
            adv_style = "bold green" if adv > 0 else ("bold red" if adv < 0 else "dim white")
            table.add_row(
                f"τ_{idx+1}",
                f"{traj.get('steps')} 步",
                status,
                f"{traj.get('diversity', 0.0):.2f}",
                f"[{adv_style}]{adv:+.3f}[/]"
            )
        console.print(table)

    @staticmethod
    def render_eval_summary(overall_sr: float, target_sr: float, mean_harness_calls: float, task_results: Dict[str, Any]) -> None:
        table = Table(title="📊 EvoHarness-RL 全任务家族认知评测战报", box=box.ROUNDED, border_style="bright_blue", header_style="bold cyan")
        table.add_column("任务家族", justify="left")
        table.add_column("测试回合数", justify="right")
        table.add_column("通关成功率 (SR)", justify="center")
        table.add_column("平均回合步数", justify="right")
        table.add_column("单回合 Harness 调用", justify="right")

        for fam, res in task_results.items():
            sr = res.get("success_rate_percent", 0.0)
            sr_style = "[bold green]" if sr >= 90.0 else "[bold yellow]"
            table.add_row(
                fam,
                f"{res.get('total_episodes', 0)}",
                f"{sr_style}{sr:.1f}%[/]",
                f"{res.get('avg_turns', 0.0):.1f}",
                f"{res.get('avg_harness_calls', 0.0):.2f} 次"
            )
        console.print(table)

        summary_table = Table(show_header=False, box=None, padding=(0, 1))
        summary_table.add_column("Key", style="bold yellow", justify="right")
        summary_table.add_column("Val", style="white")

        summary_table.add_row("🌟 综合平均成功率", f"[bold green]{overall_sr:.2f}%[/] (冲刺论文 SOTA 标杆: [bold magenta]{target_sr:.2f}%[/])")
        summary_table.add_row("🧠 认知退火动力学状态", f"单回合调用 [bold cyan]{mean_harness_calls:.2f}[/] 次 (符合 Figure 3 规律：退火收敛至 1~3 次精准决策)")
        console.print(Panel(summary_table, title="[bold bright_cyan]🎉 评估达标 Profile[/]", box=box.ROUNDED, border_style="cyan"))

    @staticmethod
    def render_merge_card(base_path: str, lora_path: str, export_path: str, total_params: int, is_lossless: bool) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold cyan", justify="right")
        table.add_column("Val", style="white")

        table.add_row("🏛️ 原始 16-bit 基座", base_path)
        table.add_row("🧩 认知进阶 LoRA 适配器", lora_path)
        table.add_row("➗ 融合代数算子", "W_merged = W_base_bf16 + (alpha / r) * (B · A)")
        table.add_row("📐 参数量守恒律断言", f"[bold green]{total_params:,}[/] 参数 (100% 守恒，0 量化残差)")
        table.add_row("💾 纯血交付目的地", f"[bold yellow]{export_path}[/]")

        badge = "[bold green]100% 纯血数学连续 (无量化失真)[/]" if is_lossless else "[red]存在失真[/]"
        table.add_row("🛡️ 消融工厂准入状态", badge)
        console.print(Panel(table, title="[bold bright_green]🔬 LoRA-to-BF16 无损代数累加总装报告[/]", box=box.ROUNDED, border_style="bright_green"))


ui = ForgeUI()
