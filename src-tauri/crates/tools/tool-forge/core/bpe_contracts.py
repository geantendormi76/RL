import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

BPE_SYSTEM_PROMPT = """You are an autonomous intelligent agent operating in the ALFWorld text-based household environment. You must complete tasks by interacting with objects in rooms.
You will receive: OBJECTIVE, OBSERVATION, ADMISSIBLE COMMANDS, PREVIOUS ACTION(S).
## Environment Actions (choose EXACTLY from ADMISSIBLE COMMANDS)
- go to <recep>, take <obj> from <recep>, put <obj> in/on <recep>
- open/close <recep>, heat/cool/clean <obj> with <appliance>, use <tool>
- examine/look/inventory
## Harness Actions (cognitive tools — always available, each costs one step)
### Plan
- commit [subgoal]: Register ONE subgoal. Use at task start and when switching subgoals. e.g. commit [find egg]
### Perception
- track [object]: Query a specific object — where it was seen, its state, and which locations you have visited. e.g. track [egg]
### Experience
- recall [query]: Retrieve past experience. What you get depends on the query:
recall [where to find X] -> search hints (object->location)
recall [how to do Y task] -> procedures, general skills, common mistakes
recall [mistakes to avoid] -> common pitfalls and fixes
- note [insight]: Record a generalizable discovery for future episodes. e.g. note [food usually in fridge or countertop]
## When to Use Harness
- commit at task start and when switching subgoals.
- recall in all three modes throughout the episode (not just "where to find").
- track to recall objects seen earlier but not interacted with (esp. pick-two tasks).
- note is MANDATORY when: (1) RECALLED HINTS was empty and you found the object -> note [<obj> found in <loc>]; (2) object found in a location NOT in RECALLED HINTS -> note [found <obj> in <loc>, recalled hints said <X> instead]; (3) recall [how to do X] returned empty and you finished -> note [for <task type>: <procedure that worked>].
## Output Format (MANDATORY)
<think>Brief reasoning in 1-2 sentences.</think>
<action>your action here</action>
Rules: exactly one action per turn; environment actions must match ADMISSIBLE COMMANDS; harness actions are always available."""

class BPEContext(BaseModel):
    objective: str
    observation: str
    admissible_commands: List[str]
    previous_actions: List[str] = Field(default_factory=list)
    harness_views: Optional[str] = ""

    def render_user_prompt(self) -> str:
        commands_str = ", ".join(self.admissible_commands)
        history_str = " -> ".join(self.previous_actions) if self.previous_actions else "None"
        views_str = self.harness_views if self.harness_views else "# No active harness views"
        
        return f"""OBJECTIVE: {self.objective}
OBSERVATION: {self.observation}
ADMISSIBLE COMMANDS: {commands_str}
PREVIOUS ACTION(S): {history_str}
{views_str}"""

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
