import os
import sys
import json
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.forge_config import global_config
from core.bpe_contracts import BPE_SYSTEM_PROMPT, BPEContext, BPEResponse
from core.bpe_engine import BPEEngine

def build_expert_trajectories() -> List[Dict]:
    tasks = [
        {
            "family": "clean",
            "objective": "clean some kettle and put it in diningtable.",
            "steps": [
                {
                    "obs": "You are in the middle of a normal kitchen. Looking around you see a countertop 1, a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a diningtable 1.",
                    "cmds": ["go to countertop 1", "go to sinkbasin 1", "go to stoveburner 1", "go to stoveburner 2", "go to stoveburner 3", "go to diningtable 1"],
                    "think": "Register the first subgoal before acting; decompose the task into locate, clean, and place.",
                    "action": "commit [find kettle]"
                },
                {
                    "obs": "You are in the kitchen. Plan is registered.",
                    "cmds": ["go to countertop 1", "go to sinkbasin 1", "go to stoveburner 1", "go to stoveburner 2", "go to stoveburner 3", "go to diningtable 1"],
                    "think": "Recall both the general clean-task procedure and likely kettle locations.",
                    "action": "recall [where to find kettle]"
                },
                {
                    "obs": "You are facing the kitchen. Countertop 1 is empty; stoveburner 3 has a kettle 1.",
                    "cmds": ["go to countertop 1", "go to stoveburner 3", "go to sinkbasin 1", "go to diningtable 1"],
                    "think": "The prior indicates kettle is often on stoveburner or countertop. Move to stoveburner 3.",
                    "action": "go to stoveburner 3"
                },
                {
                    "obs": "On the stoveburner 3, you see a kettle 1.",
                    "cmds": ["take kettle 1 from stoveburner 3", "go to countertop 1", "go to sinkbasin 1"],
                    "think": "Kettle is found. Take the kettle to proceed to cleaning.",
                    "action": "take kettle 1 from stoveburner 3"
                },
                {
                    "obs": "You take the kettle 1 from the stoveburner 3.",
                    "cmds": ["go to sinkbasin 1", "go to diningtable 1", "clean kettle 1 with sinkbasin 1"],
                    "think": "Record successful discovery in note buffer and commit to next subgoal: cleaning.",
                    "action": "note [kettle found at stoveburner 3]"
                },
                {
                    "obs": "Note buffered. You are holding kettle 1.",
                    "cmds": ["go to sinkbasin 1", "go to diningtable 1"],
                    "think": "Commit the clean subgoal.",
                    "action": "commit [clean kettle with sinkbasin]"
                },
                {
                    "obs": "Plan updated. You are holding kettle 1.",
                    "cmds": ["go to sinkbasin 1", "go to diningtable 1"],
                    "think": "Move to sinkbasin to clean the kettle.",
                    "action": "go to sinkbasin 1"
                },
                {
                    "obs": "You arrive at sinkbasin 1.",
                    "cmds": ["clean kettle 1 with sinkbasin 1", "go to diningtable 1"],
                    "think": "Clean the kettle using the sinkbasin.",
                    "action": "clean kettle 1 with sinkbasin 1"
                },
                {
                    "obs": "You clean the kettle 1 using the sinkbasin 1.",
                    "cmds": ["go to diningtable 1", "commit [place kettle in diningtable]"],
                    "think": "Commit to the final placement subgoal.",
                    "action": "commit [place kettle on diningtable]"
                },
                {
                    "obs": "Plan updated. You are holding clean kettle 1.",
                    "cmds": ["go to diningtable 1", "go to countertop 1"],
                    "think": "Move to diningtable 1 to complete placement.",
                    "action": "go to diningtable 1"
                },
                {
                    "obs": "You arrive at diningtable 1.",
                    "cmds": ["put kettle 1 in/on diningtable 1"],
                    "think": "Place the cleaned kettle on the dining table.",
                    "action": "put kettle 1 in/on diningtable 1"
                }
            ]
        },
        {
            "family": "heat",
            "objective": "heat some mug and put it on coffeemachine 1.",
            "steps": [
                {
                    "obs": "You see a countertop 1, a microwave 1, a coffeemachine 1, a cabinet 1.",
                    "cmds": ["go to countertop 1", "go to microwave 1", "go to coffeemachine 1", "go to cabinet 1"],
                    "think": "Commit the first subgoal to locate the mug.",
                    "action": "commit [find mug]"
                },
                {
                    "obs": "Plan committed. Search for mug.",
                    "cmds": ["go to countertop 1", "go to microwave 1", "go to coffeemachine 1", "go to cabinet 1"],
                    "think": "Recall likely locations for mugs.",
                    "action": "recall [where to find mug]"
                },
                {
                    "obs": "You recall mug is often on countertop or cabinet.",
                    "cmds": ["go to countertop 1", "go to cabinet 1"],
                    "think": "Check countertop 1 first.",
                    "action": "go to countertop 1"
                },
                {
                    "obs": "On the countertop 1, you see a mug 1.",
                    "cmds": ["take mug 1 from countertop 1", "go to microwave 1"],
                    "think": "Take mug 1.",
                    "action": "take mug 1 from countertop 1"
                },
                {
                    "obs": "You take mug 1 from countertop 1.",
                    "cmds": ["go to microwave 1", "go to coffeemachine 1"],
                    "think": "Commit the heating subgoal.",
                    "action": "commit [heat mug with microwave]"
                },
                {
                    "obs": "Plan updated. You are holding mug 1.",
                    "cmds": ["go to microwave 1", "go to coffeemachine 1"],
                    "think": "Move to microwave.",
                    "action": "go to microwave 1"
                },
                {
                    "obs": "You arrive at microwave 1.",
                    "cmds": ["heat mug 1 with microwave 1", "go to coffeemachine 1"],
                    "think": "Heat the mug using the microwave.",
                    "action": "heat mug 1 with microwave 1"
                },
                {
                    "obs": "You heat the mug 1 using the microwave 1.",
                    "cmds": ["go to coffeemachine 1", "commit [place mug on coffeemachine]"],
                    "think": "Commit the final place subgoal.",
                    "action": "commit [place mug on coffeemachine]"
                },
                {
                    "obs": "Plan updated. Mug is hot.",
                    "cmds": ["go to coffeemachine 1", "go to countertop 1"],
                    "think": "Move to coffeemachine 1.",
                    "action": "go to coffeemachine 1"
                },
                {
                    "obs": "You arrive at coffeemachine 1.",
                    "cmds": ["put mug 1 in/on coffeemachine 1"],
                    "think": "Place heated mug on coffeemachine 1.",
                    "action": "put mug 1 in/on coffeemachine 1"
                }
            ]
        },
        {
            "family": "pick-two",
            "objective": "put two pens on desk 1.",
            "steps": [
                {
                    "obs": "You see a desk 1, a drawer 1, a shelf 1, a sidetable 1.",
                    "cmds": ["go to desk 1", "go to drawer 1", "go to shelf 1", "go to sidetable 1"],
                    "think": "Commit subgoal to locate the first pen.",
                    "action": "commit [find first pen]"
                },
                {
                    "obs": "Plan registered.",
                    "cmds": ["go to desk 1", "go to drawer 1", "go to shelf 1", "go to sidetable 1"],
                    "think": "Query belief to check if any pen was already detected.",
                    "action": "track [pen]"
                },
                {
                    "obs": "Belief reports pen location unknown.",
                    "cmds": ["go to shelf 1", "go to drawer 1"],
                    "think": "Move to shelf 1.",
                    "action": "go to shelf 1"
                },
                {
                    "obs": "On shelf 1 you see pen 1 and pen 2.",
                    "cmds": ["take pen 1 from shelf 1", "take pen 2 from shelf 1"],
                    "think": "Take the first pen.",
                    "action": "take pen 1 from shelf 1"
                },
                {
                    "obs": "You take pen 1.",
                    "cmds": ["go to desk 1", "take pen 2 from shelf 1"],
                    "think": "Commit placing first pen on desk.",
                    "action": "commit [put first pen on desk]"
                },
                {
                    "obs": "Plan updated.",
                    "cmds": ["go to desk 1", "go to shelf 1"],
                    "think": "Move to desk 1.",
                    "action": "go to desk 1"
                },
                {
                    "obs": "You arrive at desk 1.",
                    "cmds": ["put pen 1 in/on desk 1"],
                    "think": "Place pen 1 on desk 1.",
                    "action": "put pen 1 in/on desk 1"
                },
                {
                    "obs": "You put pen 1 on desk 1.",
                    "cmds": ["commit [find second pen]", "go to shelf 1"],
                    "think": "Use track to recall where the second pen was seen.",
                    "action": "track [pen 2]"
                },
                {
                    "obs": "Track reports pen 2 is at shelf 1.",
                    "cmds": ["go to shelf 1", "commit [fetch pen 2 from shelf 1]"],
                    "think": "Commit to fetching pen 2 from shelf 1.",
                    "action": "commit [fetch pen 2 from shelf 1]"
                },
                {
                    "obs": "Plan updated.",
                    "cmds": ["go to shelf 1", "go to desk 1"],
                    "think": "Go back to shelf 1.",
                    "action": "go to shelf 1"
                },
                {
                    "obs": "On shelf 1 you see pen 2.",
                    "cmds": ["take pen 2 from shelf 1"],
                    "think": "Take pen 2.",
                    "action": "take pen 2 from shelf 1"
                },
                {
                    "obs": "You take pen 2.",
                    "cmds": ["go to desk 1"],
                    "think": "Go to desk 1 to place second pen.",
                    "action": "go to desk 1"
                },
                {
                    "obs": "You arrive at desk 1.",
                    "cmds": ["put pen 2 in/on desk 1"],
                    "think": "Put pen 2 on desk 1 to finish task.",
                    "action": "put pen 2 in/on desk 1"
                }
            ]
        },
        {
            "family": "pick-place",
            "objective": "put a book on sofa 1.",
            "steps": [
                {
                    "obs": "You see an armchair 1, a sidetable 1, a sofa 1.",
                    "cmds": ["go to armchair 1", "go to sidetable 1", "go to sofa 1"],
                    "think": "Commit initial subgoal to find book.",
                    "action": "commit [find book]"
                },
                {
                    "obs": "Plan committed.",
                    "cmds": ["go to armchair 1", "go to sidetable 1", "go to sofa 1"],
                    "think": "Recall likely place for book.",
                    "action": "recall [where to find book]"
                },
                {
                    "obs": "Recall reports books commonly on sidetable or shelf.",
                    "cmds": ["go to sidetable 1", "go to armchair 1"],
                    "think": "Move to sidetable 1.",
                    "action": "go to sidetable 1"
                },
                {
                    "obs": "On sidetable 1 you see a book 1.",
                    "cmds": ["take book 1 from sidetable 1"],
                    "think": "Take book 1.",
                    "action": "take book 1 from sidetable 1"
                },
                {
                    "obs": "You take book 1.",
                    "cmds": ["go to sofa 1", "commit [place book on sofa]"],
                    "think": "Commit place subgoal.",
                    "action": "commit [place book on sofa]"
                },
                {
                    "obs": "Plan updated.",
                    "cmds": ["go to sofa 1", "go to sidetable 1"],
                    "think": "Move to sofa 1.",
                    "action": "go to sofa 1"
                },
                {
                    "obs": "You arrive at sofa 1.",
                    "cmds": ["put book 1 in/on sofa 1"],
                    "think": "Put book 1 on sofa 1.",
                    "action": "put book 1 in/on sofa 1"
                }
            ]
        },
        {
            "family": "cool",
            "objective": "cool some apple and put it on countertop 1.",
            "steps": [
                {
                    "obs": "You see a fridge 1, a countertop 1, a diningtable 1.",
                    "cmds": ["go to fridge 1", "go to countertop 1", "go to diningtable 1"],
                    "think": "Commit find apple subgoal.",
                    "action": "commit [find apple]"
                },
                {
                    "obs": "Plan committed.",
                    "cmds": ["go to fridge 1", "go to countertop 1", "go to diningtable 1"],
                    "think": "Recall where apple is usually found.",
                    "action": "recall [where to find apple]"
                },
                {
                    "obs": "Recall indicates apple often on diningtable or countertop.",
                    "cmds": ["go to diningtable 1", "go to countertop 1"],
                    "think": "Check diningtable 1.",
                    "action": "go to diningtable 1"
                },
                {
                    "obs": "On diningtable 1 you see an apple 1.",
                    "cmds": ["take apple 1 from diningtable 1"],
                    "think": "Take apple 1.",
                    "action": "take apple 1 from diningtable 1"
                },
                {
                    "obs": "You take apple 1.",
                    "cmds": ["go to fridge 1", "commit [cool apple in fridge]"],
                    "think": "Commit cool subgoal.",
                    "action": "commit [cool apple in fridge]"
                },
                {
                    "obs": "Plan updated.",
                    "cmds": ["go to fridge 1"],
                    "think": "Go to fridge 1.",
                    "action": "go to fridge 1"
                },
                {
                    "obs": "You arrive at fridge 1.",
                    "cmds": ["cool apple 1 with fridge 1"],
                    "think": "Cool apple 1 with fridge 1.",
                    "action": "cool apple 1 with fridge 1"
                },
                {
                    "obs": "Apple 1 is cooled.",
                    "cmds": ["go to countertop 1", "commit [place apple on countertop]"],
                    "think": "Commit place subgoal.",
                    "action": "commit [place apple on countertop]"
                },
                {
                    "obs": "Plan updated.",
                    "cmds": ["go to countertop 1"],
                    "think": "Move to countertop 1.",
                    "action": "go to countertop 1"
                },
                {
                    "obs": "You arrive at countertop 1.",
                    "cmds": ["put apple 1 in/on countertop 1"],
                    "think": "Put apple 1 on countertop 1.",
                    "action": "put apple 1 in/on countertop 1"
                }
            ]
        },
        {
            "family": "light",
            "objective": "examine some plate with desklamp 1.",
            "steps": [
                {
                    "obs": "You see a desk 1, a desklamp 1, a shelf 1, a drawer 1.",
                    "cmds": ["go to desk 1", "go to desklamp 1", "go to shelf 1", "go to drawer 1"],
                    "think": "Commit find plate subgoal.",
                    "action": "commit [find plate]"
                },
                {
                    "obs": "Plan committed.",
                    "cmds": ["go to desk 1", "go to shelf 1"],
                    "think": "Recall where plate is found.",
                    "action": "recall [where to find plate]"
                },
                {
                    "obs": "Recall points to shelf or desk.",
                    "cmds": ["go to shelf 1", "go to desk 1"],
                    "think": "Check shelf 1.",
                    "action": "go to shelf 1"
                },
                {
                    "obs": "On shelf 1 you see a plate 1.",
                    "cmds": ["take plate 1 from shelf 1"],
                    "think": "Take plate 1.",
                    "action": "take plate 1 from shelf 1"
                },
                {
                    "obs": "You take plate 1.",
                    "cmds": ["go to desklamp 1", "commit [examine plate with desklamp]"],
                    "think": "Commit examine subgoal.",
                    "action": "commit [examine plate with desklamp]"
                },
                {
                    "obs": "Plan updated.",
                    "cmds": ["go to desklamp 1"],
                    "think": "Move to desklamp 1.",
                    "action": "go to desklamp 1"
                },
                {
                    "obs": "You arrive at desklamp 1.",
                    "cmds": ["use desklamp 1"],
                    "think": "Use desklamp 1 to examine the plate.",
                    "action": "use desklamp 1"
                }
            ]
        }
    ]

    dataset_records = []
    for task in tasks:
        engine = BPEEngine()
        history_actions = []
        for step in task["steps"]:
            action = step["action"]
            is_harness, harness_ret = engine.execute_harness_action(action)
            if not is_harness:
                engine.step_environment(action, step["obs"])
            active_views = engine.render_active_views()
            ctx = BPEContext(
                objective=task["objective"],
                observation=step["obs"],
                admissible_commands=step["cmds"],
                previous_actions=list(history_actions),
                harness_views=active_views
            )
            resp = BPEResponse(think=step["think"], action=action)
            dataset_records.append({
                "task_family": task["family"],
                "messages": [
                    {"role": "system", "content": BPE_SYSTEM_PROMPT},
                    {"role": "user", "content": ctx.render_user_prompt()},
                    {"role": "assistant", "content": resp.render_target()}
                ]
            })
            history_actions.append(action)
    return dataset_records

if __name__ == "__main__":
    out_path = os.path.join(global_config.data_dir, "bpe_sft_dataset.jsonl")
    records = build_expert_trajectories()
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"SFT 数据集生成完成: {out_path}, 共 {len(records)} 条样本。")
