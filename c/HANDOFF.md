当前目标：
全面梳理并生成跨会话交接宪法文档 `HANDOFF.md`，将项目宏观蓝图、阶段一全量成果、当前物理状态、下一阶段实施路线以及 8 大实战致命避坑指南一次性固化落盘，供新会话 AI 无缝白盒接管。

---

原理与解说：

根据工业级软件工程与交接契约标准，`HANDOFF.md` 必须具备**“自包含性（Self-Contained）”**与**“零上下文接管能力（Zero-Context Onboarding）”**：
1. **全景路线图**：三工厂协同（认知微调 ➔ 物理消融 ➔ 标量流形量化）；
2. **已完成资产清单**：明确 `models/handoff_for_abliteration` 已经具备 8.19B 纯血 BF16 守恒张量，并通过全部 Rust/Python 双语言测试；
3. **当前停靠点**：阶段一（`RL`）圆满竣工，正停靠在向阶段二（`LLM_pox` 消融工厂）交付的关口；
4. **血泪避坑手册**：详细记录 C 扩展解耦、TRL 1.x 传参、GRPO 即时反向传播防爆显存、LoRA-to-BF16 纯血融合等硬核经验，彻底杜绝新会话 AI 走弯路。

---

执行内容：
请在 PowerShell 中执行以下命令，将终极交接文档落盘到项目根目录：

```powershell
Set-Location -Path "C:\dev\RL"

Set-Content -Path "HANDOFF.md" -Encoding UTF8 -Value @'
# 🛡️ 工业级全栈项目全局交接文档 (HANDOFF.md)

> **版本**：v2.0.0 (Stage 1 竣工与 Stage 2 消融交接标准版)  
> **面向对象**：在新会话中接管本项目的 AI 架构师与系统工程师。请在开始任何操作前完整阅读本文档！

---

## 1. 🎯 项目终极目标与硬件物理约束

* **终极业务目标**：在单张消费级显卡（**NVIDIA GeForce RTX 3060 12GB**）上，打造出一台物理体积仅 **~3.5GB**、具备 **0.00% 绝对零拒答率**、在长流程多工具 Agent 任务上比肩甚至超越 Claude Opus 4.5（ALFWorld 96.9% 成功率）的桌面级无限制私有化 Codex 编程与工作流小模型。
* **硬件与开发环境**：
  * OS: Windows 11 / Windows Native MSVC (`x86_64-pc-windows-msvc`)
  * GPU: NVIDIA RTX 3060 (12,288 MiB VRAM), CUDA 12.4, Driver 591.86
  * 运行时: Windows Native `uv` (Python 3.11 隔离沙箱), `rustc 1.97.1` (Cargo Workspace)
  * 显存硬顶线: 严格限制在 $\le 11,000\text{ MB}$。

---

## 2. 🏛️ 三工厂大一统流水线架构 (The Three-Factory Architecture)

全系统严格划分为三个单一职责的独立项目工厂，时序绝对不可逆：

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        【三阶段工业级总装流水线】                                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  🟢 阶段一【认知工厂】(当前项目: C:\dev\RL) ➔ [已 100% 竣工！]               │
│     * 理论：EvoHarness-RL (arXiv:2608.05446)                                           │
│     * 成果：BPE 状态机 + SFT 语法冷启动 + Cost-Aware GRPO 强化学习 + LoRA-to-BF16 无损融合│
│     * 交付产物：C:\dev\RL\models\handoff_for_abliteration (16.38 GB)       │
│                                      │ (交付纯血 BF16 浮点模型)                        │
│                                      ▼                                                 │
│  ⚪ 阶段二【消融工厂】(独立项目: C:\dev\LLM_pox) ➔ [即将执行 🚀]                        │
│     * 理论：Petrov (2026) 载波差分定理 + Joad et al. (QCRI 2026) 多维拒答单维控制阀    │
│     * 任务：载入阶段一交付模型 ➔ Position -2 提取残差流 ➔ 12~21 层行模长双投影切除      │
│     * 目标：达成 0/465 100% 绝对零拒答，D_KL < 0.0001 (GSM8K 保持率 > 93%)             │
│                                      │ (交付无审查浮点模型)                            │
│                                      ▼                                                 │
│  ⚪ 阶段三【量化工厂】(独立项目: C:\dev\GSQ_RCO) ➔ [后续执行 📦]                        │
│     * 理论：GSQ (arXiv:2604.18556) + RCO (arXiv:2605.00649)                            │
│     * 任务：二阶海森矩阵收集 ➔ Lion 优化器 5-Shift 标量搜索 ➔ 黎曼流形 2.37~3.13 bpw 分配 │
│     * 目标：压缩至 ~3.5GB 单文件 GGUF，挂载 RTX 3060 极速 150+ Tokens/s 推理           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 📦 阶段一已完成资产与白盒证据链

1. **Rust 原生多包工作空间 (`src-tauri/crates`)**：
   * `core-vram`：RAII 显存令牌守卫（锁定 11,000 MB 安全线，自动 Drop 归还，0 泄露）；
   * `core-bpe`：1:1 直译的纯血 BPE 状态机与 Prompt 契约，单元测试全部绿通 (`cargo test` OK)。
2. **Python 训练沙箱 (`src-tauri/crates/tools/tool-forge`)**：
   * `core/forge_config.py`：全局强类型配置中心；
   * `core/bpe_contracts.py`：BPE 动作空间与 `<think>...</think><action>...</action>` 契约；
   * `core/bpe_engine.py`：Belief、Progress、Experience 确定性认知状态机。
3. **5 大标准工序全部完成**：
   * `01_data` 🟢 SFT 专家轨迹题库生成完成 (`data/bpe_sft_dataset.jsonl`)；
   * `02_sft` 🟢 4-bit NF4 QLoRA 动作语义微调完成，实机探针测试 100% 闭环合规；
   * `03_grpo` 🟢 Cost-Aware GRPO 强化学习策略更新跑通，优势与真实策略梯度无偏回传；
   * `04_eval` 🟢 6 大任务家族评估套件就绪 (`outputs/eval_results.json`)；
   * `05_export` 🟢 终极 LoRA-to-BF16 矩阵代数累加融合完成，生成单文件 `model.safetensors`。
4. **终态交付模型物理验证**：
   * 路径：`C:\dev\RL\models\handoff_for_abliteration`
   * 总参数量：$8,190,735,360$ 个（8.19B 守恒），总字节数：$16,381,470,720$ 字节（15.26 GiB / 16.38 GB）纯血 `torch.bfloat16`。

---

## 4. 🚨 8 大血泪避坑手册（新会话绝对禁止再踩！）

| 编号 | 致命物理病灶 | 根本原因 | 铁律解决方案 |
| :---: | :--- | :--- | :--- |
| **01** | **Windows 下 C 扩展构建崩溃** (`jericho`) | `textworld/jericho` 在 Windows 下缺失 `cmake` 导致子进程编译中断。 | 严禁在训练沙箱引入 C 游戏库，训练沙箱只保留纯 Python 依赖，由纯 Python/Rust 模拟 BPE。 |
| **02** | **TRL 1.x 参数弃用报错** (`max_seq_length`) | TRL $\ge 0.11$ 统一规范了参数命名。 | `SFTConfig` 必须使用 `max_length` 与 `warmup_steps`，严禁使用旧版 `max_seq_length`。 |
| **03** | **PeftModel 双重注入冲突** | 外部包装 `get_peft_model` 后又向 Trainer 传递 `peft_config`。 | 保持基模裸机状态传入 Trainer，由 `SFTTrainer(model=base, peft_config=cfg)` 内部统一包装。 |
| **04** | **CoT 生成截断导致断言失败** | 探针测试生成的 `max_new_tokens` 设得太小（如 128）。 | 必须严格对齐论文 Table 4：设置 `max_new_tokens=512`，给予思考链充分展开与动作闭合空间。 |
| **05** | **标点符号污染导致检索失配** | Rust 侧使用 `split_whitespace()` 保留了冒号（`"kettle:"`）。 | 必须采用 `extract_words` 算子：基于 `!c.is_alphanumeric()` 字符切分并统一转小写。 |
| **06** | **量化噪音污染后续消融** | 4-bit 反量化伪全精度导出导致残差流噪声基底暴增。 | 必须在 CPU 上载入原始 16-bit 基座，执行 $W_{\text{base}} + \frac{\alpha}{r}(B \cdot A)$ 纯血数学矩阵加法。 |
| **07** | **GRPO 反向传播显存 OOM 爆仓** | 多个 Turn 的计算图在显存中连续累积未释放。 | 启用 `prepare_model_for_kbit_training` + 单步即时反向传播（Immediate Backward）+ `torch.cuda.empty_cache()`。 |
| **08** | **静态步骤僵化陷阱** | 将规划等同于写死的 Step 1->2->3，遇到报错闪退。 | 动态改道：BPE 支持随时接收工具报错，由 8B 大脑自主触发 `re-commit` 插入纠错规划。 |

---

## 5. 📍 当前停靠状态与下一步执行计划

* **当前停靠状态**：阶段一全部收官，所有代码、测试、模型资产已锁定。
* **下一步执行计划（阶段二：物理消融破限）**：
  1. 切换终端至消融项目工厂：`cd C:\dev\LLM_pox`；
  2. 读取本项目输出的基座：`C:\dev\RL\models\handoff_for_abliteration`；
  3. 执行 `LLM_pox v4.0` 标准工序（Unmatched 对比 ➔ 95% Winsorization ➔ Layer 12~21 行模长保持双投影）；
  4. 达成 0.00% 绝对零拒答，产出消融后的 16-bit 模型供阶段三（`GSQ_RCO`）量化！
