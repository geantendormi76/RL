import re
from typing import Dict, List, Optional, Tuple

class BeliefTracker:
    def __init__(self, edge_capacity: int = 48):
        self.edge_capacity = edge_capacity
        self.object_locations: Dict[str, str] = {}
        self.object_states: Dict[str, List[str]] = {}
        self.visited_receptacles: set = set()

    def update_from_step(self, action: str, observation: str):
        go_match = re.search(r"go to ([\w\s]+ \d+)", action)
        if go_match:
            self.visited_receptacles.add(go_match.group(1).strip())

        take_match = re.search(r"take ([\w\s]+ \d+) from ([\w\s]+ \d+)", action)
        if take_match:
            obj, recep = take_match.group(1).strip(), take_match.group(2).strip()
            self.object_locations[obj] = "held_by_agent"
            self.visited_receptacles.add(recep)

        put_match = re.search(r"put ([\w\s]+ \d+) in/on ([\w\s]+ \d+)", action)
        if put_match:
            obj, recep = put_match.group(1).strip(), put_match.group(2).strip()
            self.object_locations[obj] = recep
            self.visited_receptacles.add(recep)

        clean_match = re.search(r"clean ([\w\s]+ \d+)", action)
        if clean_match:
            obj = clean_match.group(1).strip()
            states = self.object_states.setdefault(obj, [])
            if "clean" not in states:
                states.append("clean")

        heat_match = re.search(r"heat ([\w\s]+ \d+)", action)
        if heat_match:
            obj = heat_match.group(1).strip()
            states = self.object_states.setdefault(obj, [])
            if "hot" not in states:
                states.append("hot")

        cool_match = re.search(r"cool ([\w\s]+ \d+)", action)
        if cool_match:
            obj = cool_match.group(1).strip()
            states = self.object_states.setdefault(obj, [])
            if "cool" not in states:
                states.append("cool")

        see_matches = re.findall(r"see (?:a |an )?([\w\s]+ \d+)", observation)
        if see_matches and go_match:
            current_recep = go_match.group(1).strip()
            for found_obj in see_matches:
                found_obj_clean = found_obj.strip()
                if len(self.object_locations) < self.edge_capacity:
                    self.object_locations[found_obj_clean] = current_recep

    def query(self, target: str) -> str:
        target = target.strip().lower()
        if target == "world":
            loc_summary = ", ".join([f"{k} in/on {v}" for k, v in self.object_locations.items()]) if self.object_locations else "None"
            visited_summary = ", ".join(self.visited_receptacles) if self.visited_receptacles else "None"
            return f"World Summary -> Known Objects: [{loc_summary}] | Visited Receptacles: [{visited_summary}]"
        
        matched_locs = [f"{k} is at {v}" for k, v in self.object_locations.items() if target in k.lower()]
        matched_states = [f"{k} is {', '.join(v)}" for k, v in self.object_states.items() if target in k.lower()]
        
        loc_res = "; ".join(matched_locs) if matched_locs else f"{target} location unknown"
        state_res = "; ".join(matched_states) if matched_states else f"{target} state: normal"
        visited_summary = ", ".join(self.visited_receptacles) if self.visited_receptacles else "None"
        
        return f"Track [{target}] -> {loc_res} | {state_res} | Visited: [{visited_summary}]"

class ProgressTracker:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.subgoals: List[str] = []

    def commit(self, subgoal: str) -> str:
        clean_subgoal = subgoal.strip()
        if clean_subgoal not in self.subgoals:
            if len(self.subgoals) >= self.capacity:
                self.subgoals.pop(0)
            self.subgoals.append(clean_subgoal)
        return f"Plan Committed: Step {len(self.subgoals)} -> {clean_subgoal}"

    def render_plan(self) -> str:
        if not self.subgoals:
            return "# PLAN: []"
        items = [f"Step {i+1}: {g}" for i, g in enumerate(self.subgoals)]
        return f"# PLAN: [{', '.join(items)}]"

class ExperienceStore:
    def __init__(self, category_capacity: int = 80, top_k: int = 3):
        self.capacity = category_capacity
        self.top_k = top_k
        self.skills: Dict[str, Dict[str, int]] = {
            "general": {
                "Phase-ordered plan: locate -> manipulate -> place": 1,
                "Examine container contents before moving away": 1,
            },
            "task_specific": {
                "clean task: locate object -> clean with sinkbasin -> place in receptacle": 1,
                "heat task: locate object -> heat with microwave -> place in receptacle": 1,
            },
            "mistakes": {
                "Blindly trusting stale location hints": 1,
                "Repeatedly searching the same empty location": 1,
            },
            "search_priors": {
                "kettle: often found on stoveburner or countertop": 1,
                "mug: often found in cabinet, coffeemachine or countertop": 1,
            }
        }
        self.note_buffer: List[str] = []

    def recall(self, query: str) -> str:
        query_words = set(re.findall(r"\w+", query.lower()))
        matched_results = []
        for cat_name, entries in self.skills.items():
            scored_entries = []
            for text, count in entries.items():
                entry_words = set(re.findall(r"\w+", text.lower()))
                overlap = len(query_words.intersection(entry_words))
                if overlap > 0:
                    scored_entries.append((overlap, text, count))
            scored_entries.sort(key=lambda x: (x[0], x[2]), reverse=True)
            for _, text, _ in scored_entries[:self.top_k]:
                entries[text] += 1
                matched_results.append(f"[{cat_name.upper()}] {text}")
        
        if not matched_results:
            return "Recall -> No relevant prior found."
        return "Recall -> " + " | ".join(matched_results[:self.top_k])

    def note(self, insight: str) -> str:
        clean_insight = insight.strip()
        self.note_buffer.append(clean_insight)
        return f"Note Buffered: [{clean_insight}]"

class BPEEngine:
    def __init__(self):
        self.belief = BeliefTracker()
        self.progress = ProgressTracker()
        self.experience = ExperienceStore()
        self.last_track_or_recall_view: Optional[str] = None

    def step_environment(self, action: str, observation: str):
        self.belief.update_from_step(action, observation)

    def execute_harness_action(self, action_str: str) -> Tuple[bool, str]:
        commit_match = re.search(r"commit\s*\[(.*?)\]", action_str, re.IGNORECASE)
        if commit_match:
            res = self.progress.commit(commit_match.group(1))
            self.last_track_or_recall_view = res
            return True, res

        track_match = re.search(r"track\s*\[(.*?)\]", action_str, re.IGNORECASE)
        if track_match:
            res = self.belief.query(track_match.group(1))
            self.last_track_or_recall_view = res
            return True, res

        recall_match = re.search(r"recall\s*\[(.*?)\]", action_str, re.IGNORECASE)
        if recall_match:
            res = self.experience.recall(recall_match.group(1))
            self.last_track_or_recall_view = res
            return True, res

        note_match = re.search(r"note\s*\[(.*?)\]", action_str, re.IGNORECASE)
        if note_match:
            res = self.experience.note(note_match.group(1))
            self.last_track_or_recall_view = res
            return True, res

        return False, "Not a valid harness action."

    def render_active_views(self) -> str:
        plan_str = self.progress.render_plan()
        view_str = self.last_track_or_recall_view if self.last_track_or_recall_view else "# RECALLED HINTS: None"
        return f"{plan_str}\n# ACTIVE HARNESS VIEW: {view_str}"
