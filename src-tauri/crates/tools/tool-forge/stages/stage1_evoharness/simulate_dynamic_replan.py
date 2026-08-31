import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.bpe_contracts import BPEContext
from core.bpe_engine import BPEEngine

def simulate_dynamic_replan():
    print("=" * 80)
    print("🛡️  全能 Agent 动态规划 (Dynamic Re-planning) 与多工具调度实测")
    print("=" * 80)

    engine = BPEEngine()
    history = []
    
    # 场景: 用户给出一个优化任务
    task_goal = "优化本地推理算子，要求比基线提速 20% 以上。"
    print(f"用户需求: {task_goal}\n")

    # Step 1: 初始规划 -> 先进行性能分析
    print("--- [Turn 1: 大脑初始规划] ---")
    llm_thought_1 = "面对性能优化任务，不能盲目手写代码，第一步必须先调用 profile 工具测量基线耗时。"
    llm_action_1 = "commit [调用 profiler 测量基准耗时]"
    print(f"[8B 大脑思考]: {llm_thought_1}")
    print(f"[8B 大脑动作]: {llm_action_1}")
    engine.execute_harness_action(llm_action_1)
    history.append(llm_action_1)

    # Step 2: 工具执行 -> 发现是 Memory-Bound (显存瓶颈)
    print("\n--- [Turn 2: Harness 执行 profile 工具并返回硬件反馈] ---")
    tool_output = "Profiler 反馈: 计算单元利用率仅 15%, 显存带宽跑满 98% (严重 Memory-Bound 瓶颈)！"
    print(f"[工具返回]: {tool_output}")
    
    # Step 3: 大脑动态改道 -> 绝不死板走老路，根据反馈调整规划
    print("\n--- [Turn 3: 8B 大脑根据反馈动态调整方案 (非固定步骤)] ---")
    llm_thought_2 = "既然是 Memory-Bound 瓶颈，传统增加计算量的优化无效，必须动态改道：采用 float4 向量化内存加载和算子融合减少显存读写！"
    llm_action_2 = "commit [实施 float4 向量化内存加载与算子融合]"
    print(f"[8B 大脑思考]: {llm_thought_2}")
    print(f"[8B 大脑动作 (动态覆写规划)]: {llm_action_2}")
    engine.execute_harness_action(llm_action_2)
    history.append(llm_action_2)

    # Step 4: 调工具改代码并编译
    print("\n--- [Turn 4: 大脑发出代码修改与编译指令] ---")
    llm_action_3 = "edit_file [kernels/fast_op.cu]"
    print(f"[8B 大脑动作]: {llm_action_3}")
    history.append(llm_action_3)
    
    # Step 5: 验算通过
    print("\n--- [Turn 5: 最终测速与成功交付] ---")
    final_perf = "Benchmark 结果: 耗时从 2.30ms 降低至 0.95ms，提速 2.42x (达成 20%+ 提速目标)！"
    print(f"[Harness 验算返回]: {final_perf}")
    print(engine.render_active_views())

    print("\n" + "=" * 80)
    print("🎉 仿真完成！这就是真实落地的 Agent 架构：")
    print("8B 负责灵活决策与动态改道，BPE 负责状态同步，Tools 负责在具体环境中真实落地！\n")

if __name__ == "__main__":
    simulate_dynamic_replan()
