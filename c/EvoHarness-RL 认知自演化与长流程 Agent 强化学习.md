# 🛡️ EvoHarness-RL 认知自演化与长流程 Agent 强化学习工程规格书 (v2.0.0 工业级大一统版)

> **版本**：v2.0.0 (SFT 语法冷启动 + GRPO 策略梯度自进化 + 全域多工具调度大一统版)  
> **定位**：面向大语言模型（LLMs）长流程复杂任务规划、自演化认知脚手架、强化学习策略调优与多工具 Agent 调度的通用工业级工程标准与数学算法规格书。  
> **核心范式**：BPE（Belief, Progress, Experience）动态认知状态机 + Cost-Aware GRPO 强化学习 + 动作语义 SFT 冷启动 + LoRA 增量继承进化 + LoRA-to-BF16 零量化失真无损融合。  
> **理论依据**：
> 1. Ning et al. (UIUC & Meta AI 2026) *EvoHarness-RL: Learning Self-Evolving Runtime Harness for Long-Horizon LLM Agents* (arXiv:2608.05446)
> 2. Dai et al. (Tsinghua AIR & ByteDance Seed 2026) *CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation* (arXiv:2602.24286)
> **设计哲学**：钱学森系统工程论 —— 抽象外部动态马具（Harness），通过强化学习让 8B 决策大脑内化元认知策略（Harness Annealing），以单张消费级显卡（RTX 3060 12GB）斩获超越 Claude Opus 4.5 旗舰级长流程成功率（96.9%）。

---

## 0. 最高架构总览与执行流水线状态机

EvoHarness-RL 认知流水线严格划分为 **5 道单向推进、单步可验证的标准工序**：

```text
                           [ 输入: 原始开源基座 (如 Qwen3-8B 官方权重) ]
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ 【工序 01】SFT 专家轨迹与 BPE 样本构建 (01_prepare_sft_dataset.py)                             │
│  - 载入 BPE 认知状态机 (Belief / Progress / Experience)，覆盖 6 大任务家族                   │
│  - 收集/生成高质量通关轨迹，提取单步决策对并注入 <think>...</think><action>...</action> 契约  │
│  - 确保认知元动作 (commit, track, recall, note) 调用占比对齐黄金比例 (~18% - 40%)             │
│  - 产出物：data/bpe_sft_dataset.jsonl                                                         │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ 【工序 02】SFT 认知动作语义微调 (02_train_sft.py)                                             │
│  - 4-bit NF4 显存锁定载入基座 + 全线性层 LoRA 挂载 (q/k/v/o_proj, gate/up/down_proj)          │
│  - 启用梯度检查点 + 8-bit Paged AdamW 分页优化器 (RTX 3060 12GB 显存峰值 <= 7.8GB)            │
│  - 训练 3 个 Epoch，让模型牢固内化 BPE 动作语义反射与思考格式                                 │
│  - 产出物：models/evoharness_sft (BF16 LoRA 基础适配器)                                       │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ 【工序 03】Cost-Aware GRPO 强化学习策略自进化 (03_train_grpo.py)                              │
│  - 初始化自 SFT LoRA Checkpoint (继承已有语法知识，继续微调同一个 LoRA 权重)                  │
│  - 组采样机制 (Group Sampling G=4~8): 对同一任务头脑风暴多条路线，消除 Critic 网络 (0MB 显存)  │
│  - 五维复合奖励驱动: R = R_succ + λ_eff*R_eff + λ_div(u)*R_div - λ_spam*R_spam - λ_inv*R_inv  │
│  - 组内相对优势归一化: A_i = (R_i - mean(R)) / (std(R) + eps) ➔ 真实 Token 策略梯度反向传播    │
│  - 驱动认知退火 (Harness Annealing): 动作调用自发从频繁查询收敛为单回合 1~3 次关键决策        │
│  - 产出物：models/evoharness_grpo (集大成的进阶 LoRA 适配器)                                  │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ 【工序 04】BPE Agent 成功率评估与退火动力学分析 (04_eval_agent_bpe.py)                        │
│  - 针对 6 大任务家族执行长流程多回合交互评测，断言综合成功率 >= 95% (冲刺 96.9%)              │
│  - 统计 Harness 认知工具调用均值与分布，监控 LFU 经验池紧凑度与淘汰健康度                    │
│  - 产出物：outputs/eval_results.json                                                          │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ 【工序 05】LoRA-to-BF16 无损模型融合与消融交付 (05_export_for_abliteration.py)                │
│  - 在 CPU 内存载入未量化的原始 16-bit 基座 (torch.bfloat16)                                   │
│  - 实施 W_merged = W_original_bf16 + (alpha/r)*(B·A) 逐比特代数累加 (参数量完全守恒 8.19B)   │
│  - 导出 100% 纯血 16-bit Safetensors 模型 (无量化噪音、各向同性平滑分布)                      │
│  - 产出物：models/handoff_for_abliteration (直接交付 LLM_pox 消融工厂)                        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 核心数学理论与算法白盒直译

### 1.1 BPE 统一动态认知状态空间抽象（Dynamic Harness Substrate）

在长流程交互中，外部状态统一形式化为三元组：
$$H_t = (B_t, P_t, E_t)$$

1. **信念状态（Belief $B_t$）**：
   维护环境物理常识与状态图谱（如“文件路径、函数位置、编译报错、物体位置”），容量上限 48 边。通过规则解析器从工具输出中提取，支持 `track [target]` 精确查询与 `track [world]` 全局概览；
2. **进度状态（Progress $P_t$）**：
   维护显式任务分解与子目标状态序列 $(g_i, \sigma_i)$，容量上限 8 个子目标。通过 `commit [subgoal]` 注册当前阶段。**支持动态改道（Dynamic Re-planning）**：当遇到环境异常时，模型可自主覆写或插入新子目标进行自愈；
3. **经验状态（Experience $E_t$）**：
   跨回合知识库，划分为通用策略（General）、任务专精（Task-specific）、避坑指南（Mistakes）与搜索先验（Search Priors）四大类目。通过关键词重合度检索 Top-3，并采用最不经常使用（LFU）策略淘汰旧知识。

### 1.2 全域通用 Agent 架构大一统定理

$$\text{Agent} = \text{Policy (大脑)} + \text{Harness (马具中枢)} + \text{Tools (执行能力)} + \text{Environment (物理环境)}$$

* **8B Policy 大脑**：唯一负责思考“下一步该干什么？”，输出 `<think>...</think><action>...</action>`；
* **Harness 马具（Rust core-bpe）**：负责维护上下文、拼接 BPE 视窗、截获 Action 并派发；
* **Tools 工具集（可插拔）**：
  * CUDA 场景：`profile.py`, `compile.sh`, `verify.py`；
  * Codex 场景：`read_file`, `edit_file`, `cargo_test`；
  * 文档场景：`parse_pdf`, `read_excel`, `write_report`。

### 1.3 Cost-Aware GRPO 强化学习数学目标与无 Critic 原理

#### (1) 组内相对优势（Group Relative Advantage - 0 显存基准）
针对同一个 Prompt 题目，策略采样 $G$ 条路线（$G \in \{4, 8\}$），计算各路线总得分 $R_1, \dots, R_G$：
$$A_i = \frac{R_i - \text{mean}(\{R_1, \dots, R_G\})}{\text{std}(\{R_1, \dots, R_G\}) + \varepsilon}$$

* **零方差保护定理**：若 $R_1 = R_2 = \dots = R_G$（全胜或全败），则 $\text{std}=0 \implies A_i=0.0 \implies \mathcal{L}_{\text{PG}}=0.0000$。算法自动避免在无辨识度的样本上盲目更新参数。

#### (2) 五维复合奖励公式 (Equation 5 & 6)
$$R(\tau) = \underbrace{R_{\text{succ}}(\tau)}_{\text{结果硬门禁}} + \underbrace{\lambda_{\text{eff}} R_{\text{eff}}(\tau)}_{\text{步数效率奖金}} + \underbrace{\lambda_{\text{div}}(u) R_{\text{div}}(\tau)}_{\text{动词多样性退火}} - \underbrace{\lambda_{\text{spam}} R_{\text{spam}}(\tau)}_{\text{复读惩罚}} - \underbrace{\lambda_{\text{inv}} R_{\text{inv}}(\tau)}_{\text{格式惩罚}}$$

1. **结果硬门禁**：$R_{\text{succ}}(\tau) = 10 \cdot \mathbb{I}[\text{solved}]$（未成功完成一律 0 分）；
2. **步数效率奖金**：$R_{\text{eff}}(\tau) = \max\left(0, 1 - \frac{|\tau|}{T_{\text{max}}}\right)$（仅在成功时发放，倒逼模型走最短路径）；
3. **动词多样性探索激励（余弦退火）**：
   $$R_{\text{div}}(\tau) = \frac{|\{\text{verb}(a_t) : a_t \in \tau\}|}{|\tau|}, \quad \lambda_{\text{div}}(u) = \frac{\lambda_{\text{div}}^{\text{max}}}{2} \left(1 + \cos\left(\frac{\pi u}{U}\right)\right)$$
   * 严格考核动词词元，初期逼迫模型尝试所有认知与物理工具，后期 $\lambda_{\text{div}} \to 0$ 强制内化极简直觉；
4. **惩罚项**：连续相同动作惩罚 $R_{\text{spam}}$（上限 10 分）与格式残缺惩罚 $R_{\text{inv}}$。

#### (3) 真实策略梯度损失计算 (Policy Gradient Loss)
$$\mathcal{L}_{\text{GRPO}} = - \frac{1}{G} \sum_{i=1}^G \left[ A_i \cdot \sum_{t=1}^{|\tau_i|} \log P_\theta(a_{i,t} \mid s_{i,t}) \right]$$

### 1.4 LoRA-to-BF16 代数累加定理（Additive Merge & Shape Invariance）

LoRA 融合是**高精度矩阵代数累加，不是参数覆盖**：
$$W_{\text{merged}} = W_{\text{base}}^{\text{BF16}} + \frac{\alpha}{r} (B^{\text{BF16}} \cdot A^{\text{BF16}})$$
* **参数守恒律**：$[d_{\text{out}}, d_{\text{in}}] + [d_{\text{out}}, d_{\text{in}}] = [d_{\text{out}}, d_{\text{in}}]$，总参数量严格保持 8,190,735,360 个（8.19B），体积保持 16.38 GB；
* **消融相容性**：融合后的张量为纯血连续浮点，行范数 $\|W\|_{\text{row}}$ 各向同性平滑，为阶段二 `LLM_pox` 的行模长保持双投影算子提供完美的数学底座。

---

## 2. 核心数据契约与接口规范

### 2.1 BPE 动作与响应双向契约 (`bpe_contracts.py` & `core-bpe`)

```python
class BPEResponse(BaseModel):
    think: str
    action: str

    @classmethod
    def parse_from_text(cls, raw_text: str) -> Optional["BPEResponse"]:
        think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
        action_match = re.search(r"<action>(.*?)</action>", raw_text, re.DOTALL)
        if think_match and action_match:
            return cls(think=think_match.group(1).strip(), action=action_match.group(1).strip())
        return None

    def render_target(self) -> str:
        return f"<think>{self.think}</think><action>{self.action}</action>"
```

### 2.2 SFT 数据集格式契约 (`bpe_sft_dataset.jsonl`)

```json
{
  "task_family": "clean",
  "messages": [
    { "role": "system", "content": "You are an autonomous intelligent agent..." },
    { "role": "user", "content": "OBJECTIVE: clean some kettle and put it in diningtable.\nOBSERVATION: You are in a normal kitchen...\nADMISSIBLE COMMANDS: go to countertop 1, go to stoveburner 3...\nPREVIOUS ACTION(S): None\n# PLAN: []\n# ACTIVE HARNESS VIEW: None" },
    { "role": "assistant", "content": "<think>Register the first subgoal before acting; decompose the task into locate, clean, and place.</think><action>commit [find kettle]</action>" }
  ]
}
```

---

## 3. 工业级避坑与排查修复终极手册 (12 大致命物理病灶)

| 编号 | 致命病灶与表现 | 物理/数理根因 | 官方标准修复药方 |
| :---: | :--- | :--- | :--- |
| **01** | **Windows 下 C 扩展构建崩溃**<br>(`Failed to build jericho / textworld`) | `textworld/jericho` 依赖 C++ 编译环境，在 Windows 下因缺失 `cmake` 导致构建中断，瘫痪整个 Python 环境。 | 架构解耦：训练沙箱仅保留纯 Python 依赖（`torch+cu124`, `transformers`, `trl`, `peft`），由纯 Python / Rust 直译 BPE 引擎。 |
| **02** | **TRL 1.x 参数弃用报错**<br>(`unexpected keyword 'max_seq_length'`) | TRL 升级至 1.12+ 后，`SFTConfig` 废弃了 `max_seq_length` 与 `warmup_ratio`。 | 对齐最新 API：将 `max_seq_length` 重命名为 `max_length`，并使用 `warmup_steps` 代替 `warmup_ratio`。 |
| **03** | **PeftModel 双重注入冲突**<br>(`You passed a PeftModel instance together with a peft_config`) | 外部手动调用 `get_peft_model` 后又向 `SFTTrainer` 传入 `peft_config`，触发互斥校验。 | 保持基模裸机状态进入 Trainer，由 `SFTTrainer(model=base, peft_config=cfg)` 在内部统一完成 LoRA 适配器注入。 |
| **04** | **CoT 生成截断导致断言失败**<br>(模型只吐出一半 `<think>` 标签未闭合) | 验证探针生成的 `max_new_tokens` 设得太小（如 128），思考链尚未展开完毕即被硬件截断。 | 1:1 对齐论文 Table 4：设置 `max_new_tokens=512`，赋予思考链充足的推理与动作决策吐字空间。 |
| **05** | **标点符号污染导致检索失配**<br>(Rust `recall` 找不到 `"kettle:"` 词条) | 简单 `split_whitespace()` 保留了冒号/逗号，导致关键词交集重合度为 0。 | 采用 `extract_words` 算子：基于 `!c.is_alphanumeric()` 字符切分并统一转小写，实现纯净分词。 |
| **06** | **量化噪音污染后续消融**<br>(4-bit 反量化导出导致消融破限完全失效) | NF4 的量化残差噪音 $\epsilon_{quant}$ 淹没了残差流微弱的拒答差分势能。 | 严格执行 LoRA-to-BF16 协议：在 CPU 上载入原始 16-bit 基座，通过纯血数学矩阵加法融合适配器。 |
| **07** | **GRPO 采样方差导致策略震荡**<br>(奖励忽高忽低无法收敛) | 单步仅采样 1~2 条轨迹，离散动作高方差导致梯度混乱。 | 设置组内采样数 $G \ge 8$，并对奖励施加优势归一化 $A_i = (R_i - \bar{R}) / (\sigma_R + \varepsilon)$。 |
| **08** | **SFT 显存尖峰触发 OOM**<br>(12GB 显存溢出) | 反向传播激活值堆积与未分页优化器内存碎片。 | 开启梯度检查点（`gradient_checkpointing_enable`）+ `optim="paged_adamw_8bit"`，静态显存压制在 5.2GB。 |
| **09** | **GRPO 反向传播显存 OOM 爆仓**<br>(`CUDA out of memory during backward`) | 多轮长轨迹的前向计算图在显存中连续累积未释放，激活显存突破 26GB 虚拟上限。 | 启用 `prepare_model_for_kbit_training` + 单步即时反向传播（Immediate Backward）+ `torch.cuda.empty_cache()`，显存锁死在 7.5GB。 |
| **10** | **组内同分全零优势假死**<br>(`Advantage: [0,0,0,0], Loss: 0.0000`) | 步数上限过紧（如卡在 4 步），导致所有路线同时超时失败，组内方差为 0。 | 识别算法自保护：放宽步数预算至 $T_{\text{max}} \ge 12 \sim 70$，让优势梯度自发涌现。 |
| **11** | **Pydantic 序列化方法缺失**<br>(`AttributeError: parse_from_text`) | Python 契约类未实现文本正则提取反序列化方法。 | 1:1 对齐 Rust 侧实现：在 `BPEResponse` 中注入 `@classmethod parse_from_text`。 |
| **12** | **静态步骤僵化陷阱**<br>(面对环境报错直接闪退崩溃) | 将规划等同于死板的 Step 1->2->3 瀑布流，无法应对工具执行失败。 | 动态改道机制：BPE 状态机支持随时接收环境报错，由 8B 大脑动态 `re-commit` 自愈改道。 |

---

## 4. TDD 测试验收与双级裁判规范

### 4.1 一级裁判：数学几何与状态机逻辑断言（Unit Test）
1. **BPE 状态机构造断言**：`commit`, `track`, `recall`, `note` 状态流转与 Harness View 渲染正确率 100%；
2. **GRPO 奖励与优势断言**：成功轨迹 $R > 10.0$，复读轨迹 $R < 0.0$，全同得分组优势严格归零（$A_i=0.0$）；
3. **LoRA-to-BF16 连续性断言**：融合后权重与 FP32 累加真值最大绝对误差为 `0.00000000e+00`，总参数量严格等于 $8,190,735,360$。

### 4.2 二级裁判：全网前向保真度与实机决策断言（System Test）
1. **格式符合度断言**：实机贪婪解码（Greedy Decoding）输出必须 100% 严格包含闭合的 `<think>...</think><action>...</action>` 标签；
2. **合法动作断言**：输出动作必须完全属于给定的候选命令或合法 BPE 元动作；
3. **动态自愈断言**：在遭遇工具报错时，必须能自主触发 `re-commit` 插入纠错规划并完成任务。

---

## 5. 通用 AI 结构编译器系统提示词规范 (AI Implementation Prompt)

未来在任何新模型（如 LLaMA-3-8B、DeepSeek-R1-8B、Mistral-7B）上实施 EvoHarness-RL 认知微调时，可直接向 AI 注入以下系统提示词：

```text
你是一个顶级认知架构师与代码迁移结构编译器。
你必须严格遵循《EvoHarness-RL 认知自演化与微调工程规格书 (v2.0.0)》：
1. 严禁自创抽象逻辑与状态机流程，严格执行 5 步状态机（01_Data -> 02_SFT -> 03_GRPO -> 04_Eval -> 05_Export）；
2. 认知抽象层：1:1 落地 BPE 状态机（Belief 上限 48 边，Progress 上限 8 子目标，Experience 4 类目 LFU 淘汰）与 4 大元动作；
3. 训练与奖励层：严格对齐五维复合奖励公式 R(τ) 与余弦退火多样性激励，在 RTX 3060 12GB 上采用 4-bit NF4 QLoRA + 梯度检查点 + 单步即时反向传播；
4. 交付协议：工序 05 必须执行 LoRA-to-BF16 纯血数学无损融合（W_base + ΔW），交付 100% 连续平滑、形状守恒的 16-bit Safetensors 模型给消融工厂；
5. 一步一验证：凡涉及代码修改必须使用 PowerShell Set-Content UTF-8 原样全量落盘，测试断言通过后再推进下一步。
```
