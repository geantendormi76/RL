use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

pub const BPE_SYSTEM_PROMPT: &str = r#"You are an autonomous intelligent agent operating in the ALFWorld text-based household environment. You must complete tasks by interacting with objects in rooms.
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
Rules: exactly one action per turn; environment actions must match ADMISSIBLE COMMANDS; harness actions are always available."#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeContext {
    pub objective: String,
    pub observation: String,
    pub admissible_commands: Vec<String>,
    pub previous_actions: Vec<String>,
    pub harness_views: Option<String>,
}

impl BpeContext {
    pub fn render_user_prompt(&self) -> String {
        let commands_str = self.admissible_commands.join(", ");
        let history_str = if self.previous_actions.is_empty() {
            "None".to_string()
        } else {
            self.previous_actions.join(" -> ")
        };
        let views_str = self.harness_views.as_deref().unwrap_or("# No active harness views");

        format!(
            "OBJECTIVE: {}\nOBSERVATION: {}\nADMISSIBLE COMMANDS: {}\nPREVIOUS ACTION(S): {}\n{}",
            self.objective, self.observation, commands_str, history_str, views_str
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeResponse {
    pub think: String,
    pub action: String,
}

impl BpeResponse {
    pub fn parse_from_text(raw_text: &str) -> Option<Self> {
        let think_start = raw_text.find("<think>")?;
        let think_end = raw_text.find("</think>")?;
        let action_start = raw_text.find("<action>")?;
        let action_end = raw_text.find("</action>")?;

        if think_start < think_end && action_start < action_end && think_end <= action_start {
            let think = raw_text[think_start + 7..think_end].trim().to_string();
            let action = raw_text[action_start + 8..action_end].trim().to_string();
            Some(Self { think, action })
        } else {
            None
        }
    }

    pub fn render_target(&self) -> String {
        format!("<think>{}</think><action>{}</action>", self.think, self.action)
    }
}

fn extract_words(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefTracker {
    pub edge_capacity: usize,
    pub object_locations: HashMap<String, String>,
    pub object_states: HashMap<String, Vec<String>>,
    pub visited_receptacles: HashSet<String>,
}

impl BeliefTracker {
    pub fn new(edge_capacity: usize) -> Self {
        Self {
            edge_capacity,
            object_locations: HashMap::new(),
            object_states: HashMap::new(),
            visited_receptacles: HashSet::new(),
        }
    }

    pub fn update_from_step(&mut self, action: &str, observation: &str) {
        let action_clean = action.trim();
        
        if let Some(pos) = action_clean.find("go to ") {
            let recep = action_clean[pos + 6..].trim().to_string();
            self.visited_receptacles.insert(recep);
        }

        if let Some(pos) = action_clean.find("take ") {
            if let Some(from_pos) = action_clean.find(" from ") {
                let obj = action_clean[pos + 5..from_pos].trim().to_string();
                let recep = action_clean[from_pos + 6..].trim().to_string();
                self.object_locations.insert(obj, "held_by_agent".to_string());
                self.visited_receptacles.insert(recep);
            }
        }

        if let Some(pos) = action_clean.find("put ") {
            if let Some(in_pos) = action_clean.find(" in/on ") {
                let obj = action_clean[pos + 4..in_pos].trim().to_string();
                let recep = action_clean[in_pos + 7..].trim().to_string();
                self.object_locations.insert(obj, recep.clone());
                self.visited_receptacles.insert(recep);
            }
        }

        if let Some(pos) = action_clean.find("clean ") {
            let obj = action_clean[pos + 6..].trim().to_string();
            let states = self.object_states.entry(obj).or_default();
            if !states.contains(&"clean".to_string()) {
                states.push("clean".to_string());
            }
        }

        if let Some(pos) = action_clean.find("heat ") {
            let obj = action_clean[pos + 5..].trim().to_string();
            let states = self.object_states.entry(obj).or_default();
            if !states.contains(&"hot".to_string()) {
                states.push("hot".to_string());
            }
        }

        if let Some(pos) = action_clean.find("cool ") {
            let obj = action_clean[pos + 5..].trim().to_string();
            let states = self.object_states.entry(obj).or_default();
            if !states.contains(&"cool".to_string()) {
                states.push("cool".to_string());
            }
        }

        if let Some(pos) = observation.find("see ") {
            let see_str = &observation[pos + 4..];
            for chunk in see_str.split(',') {
                let clean_item = chunk.replace("a ", "").replace("an ", "").trim().to_string();
                if !clean_item.is_empty() && self.object_locations.len() < self.edge_capacity {
                    if let Some(recep) = self.visited_receptacles.iter().last() {
                        self.object_locations.entry(clean_item).or_insert_with(|| recep.clone());
                    }
                }
            }
        }
    }

    pub fn query(&self, target: &str) -> String {
        let clean_target = target.trim().to_lowercase();
        if clean_target == "world" {
            let locs: Vec<String> = self.object_locations.iter().map(|(k, v)| format!("{} in/on {}", k, v)).collect();
            let visited: Vec<String> = self.visited_receptacles.iter().cloned().collect();
            return format!("World Summary -> Known Objects: [{}] | Visited Receptacles: [{}]", 
                if locs.is_empty() { "None".to_string() } else { locs.join(", ") },
                if visited.is_empty() { "None".to_string() } else { visited.join(", ") }
            );
        }

        let matched_locs: Vec<String> = self.object_locations.iter()
            .filter(|(k, _)| k.to_lowercase().contains(&clean_target))
            .map(|(k, v)| format!("{} is at {}", k, v))
            .collect();
            
        let matched_states: Vec<String> = self.object_states.iter()
            .filter(|(k, _)| k.to_lowercase().contains(&clean_target))
            .map(|(k, v)| format!("{} is {}", k, v.join(", ")))
            .collect();

        let visited: Vec<String> = self.visited_receptacles.iter().cloned().collect();

        let loc_res = if matched_locs.is_empty() { format!("{} location unknown", clean_target) } else { matched_locs.join("; ") };
        let state_res = if matched_states.is_empty() { format!("{} state: normal", clean_target) } else { matched_states.join("; ") };
        let visited_res = if visited.is_empty() { "None".to_string() } else { visited.join(", ") };

        format!("Track [{}] -> {} | {} | Visited: [{}]", clean_target, loc_res, state_res, visited_res)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressTracker {
    pub capacity: usize,
    pub subgoals: Vec<String>,
}

impl ProgressTracker {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            subgoals: Vec::new(),
        }
    }

    pub fn commit(&mut self, subgoal: &str) -> String {
        let clean = subgoal.trim().to_string();
        if !self.subgoals.contains(&clean) {
            if self.subgoals.len() >= self.capacity {
                self.subgoals.remove(0);
            }
            self.subgoals.push(clean.clone());
        }
        format!("Plan Committed: Step {} -> {}", self.subgoals.len(), clean)
    }

    pub fn render_plan(&self) -> String {
        if self.subgoals.is_empty() {
            return "# PLAN: []".to_string();
        }
        let items: Vec<String> = self.subgoals.iter().enumerate().map(|(i, g)| format!("Step {}: {}", i + 1, g)).collect();
        format!("# PLAN: [{}]", items.join(", "))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceStore {
    pub capacity: usize,
    pub top_k: usize,
    pub skills: HashMap<String, HashMap<String, usize>>,
    pub note_buffer: Vec<String>,
}

impl ExperienceStore {
    pub fn new(capacity: usize, top_k: usize) -> Self {
        let mut skills = HashMap::new();
        
        let mut general = HashMap::new();
        general.insert("Phase-ordered plan: locate -> manipulate -> place".to_string(), 1);
        general.insert("Examine container contents before moving away".to_string(), 1);
        skills.insert("general".to_string(), general);

        let mut task_specific = HashMap::new();
        task_specific.insert("clean task: locate object -> clean with sinkbasin -> place in receptacle".to_string(), 1);
        skills.insert("task_specific".to_string(), task_specific);

        let mut mistakes = HashMap::new();
        mistakes.insert("Blindly trusting stale location hints".to_string(), 1);
        skills.insert("mistakes".to_string(), mistakes);

        let mut search_priors = HashMap::new();
        search_priors.insert("kettle: often found on stoveburner or countertop".to_string(), 1);
        skills.insert("search_priors".to_string(), search_priors);

        Self {
            capacity,
            top_k,
            skills,
            note_buffer: Vec::new(),
        }
    }

    pub fn recall(&mut self, query: &str) -> String {
        let query_words = extract_words(query);
        let mut all_scored: Vec<(usize, String, String, usize)> = Vec::new();

        for (cat_name, entries) in self.skills.iter() {
            for (text, count) in entries.iter() {
                let text_words = extract_words(text);
                let overlap = query_words.intersection(&text_words).count();
                if overlap > 0 {
                    all_scored.push((overlap, cat_name.clone(), text.clone(), *count));
                }
            }
        }

        all_scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.3.cmp(&a.3)));

        if all_scored.is_empty() {
            return "Recall -> No relevant prior found.".to_string();
        }

        let mut results = Vec::new();
        for (_, cat_name, text, _) in all_scored.iter().take(self.top_k) {
            if let Some(cat_map) = self.skills.get_mut(cat_name) {
                if let Some(c) = cat_map.get_mut(text) {
                    *c += 1;
                }
            }
            results.push(format!("[{}] {}", cat_name.to_uppercase(), text));
        }

        format!("Recall -> {}", results.join(" | "))
    }

    pub fn note(&mut self, insight: &str) -> String {
        let clean = insight.trim().to_string();
        self.note_buffer.push(clean.clone());
        format!("Note Buffered: [{}]", clean)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeEngine {
    pub belief: BeliefTracker,
    pub progress: ProgressTracker,
    pub experience: ExperienceStore,
    pub last_view: Option<String>,
}

impl BpeEngine {
    pub fn new() -> Self {
        Self {
            belief: BeliefTracker::new(48),
            progress: ProgressTracker::new(8),
            experience: ExperienceStore::new(80, 3),
            last_view: None,
        }
    }

    pub fn step_environment(&mut self, action: &str, observation: &str) {
        self.belief.update_from_step(action, observation);
    }

    pub fn execute_harness_action(&mut self, action: &str) -> (bool, String) {
        let action_clean = action.trim();

        if let Some(start) = action_clean.find("commit [") {
            if let Some(end) = action_clean[start..].find(']') {
                let subgoal = &action_clean[start + 8..start + end];
                let res = self.progress.commit(subgoal);
                self.last_view = Some(res.clone());
                return (true, res);
            }
        }

        if let Some(start) = action_clean.find("track [") {
            if let Some(end) = action_clean[start..].find(']') {
                let target = &action_clean[start + 7..start + end];
                let res = self.belief.query(target);
                self.last_view = Some(res.clone());
                return (true, res);
            }
        }

        if let Some(start) = action_clean.find("recall [") {
            if let Some(end) = action_clean[start..].find(']') {
                let query = &action_clean[start + 8..start + end];
                let res = self.experience.recall(query);
                self.last_view = Some(res.clone());
                return (true, res);
            }
        }

        if let Some(start) = action_clean.find("note [") {
            if let Some(end) = action_clean[start..].find(']') {
                let insight = &action_clean[start + 6..start + end];
                let res = self.experience.note(insight);
                self.last_view = Some(res.clone());
                return (true, res);
            }
        }

        (false, "Not a valid harness action.".to_string())
    }

    pub fn render_active_views(&self) -> String {
        let plan = self.progress.render_plan();
        let view = self.last_view.as_deref().unwrap_or("# RECALLED HINTS: None");
        format!("{}\n# ACTIVE HARNESS VIEW: {}", plan, view)
    }
}

impl Default for BpeEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_contracts_rendering_and_parsing() {
        let ctx = BpeContext {
            objective: "clean some kettle".to_string(),
            observation: "kitchen view".to_string(),
            admissible_commands: vec!["go to countertop 1".to_string()],
            previous_actions: vec![],
            harness_views: Some("# PLAN: []".to_string()),
        };
        let prompt = ctx.render_user_prompt();
        assert!(prompt.contains("OBJECTIVE: clean some kettle"));

        let raw_llm_output = "<think>I need to find the kettle first.</think><action>commit [find kettle]</action>";
        let parsed = BpeResponse::parse_from_text(raw_llm_output).expect("解析失败");
        assert_eq!(parsed.think, "I need to find the kettle first.");
        assert_eq!(parsed.action, "commit [find kettle]");
        assert_eq!(parsed.render_target(), raw_llm_output);
    }

    #[test]
    fn test_bpe_full_harness_lifecycle() {
        let mut engine = BpeEngine::new();

        let (ok, res) = engine.execute_harness_action("commit [find kettle]");
        assert!(ok);
        assert_eq!(res, "Plan Committed: Step 1 -> find kettle");

        let (ok, res) = engine.execute_harness_action("recall [where to find kettle]");
        assert!(ok);
        assert!(res.contains("stoveburner or countertop"));

        engine.step_environment("go to stoveburner 3", "You see a kettle 1 on stoveburner 3.");
        let (ok, res) = engine.execute_harness_action("track [kettle]");
        assert!(ok);
        assert!(res.contains("stoveburner 3"));

        let (ok, res) = engine.execute_harness_action("note [kettle found at stoveburner 3]");
        assert!(ok);
        assert_eq!(res, "Note Buffered: [kettle found at stoveburner 3]");

        let view = engine.render_active_views();
        assert!(view.contains("# PLAN: [Step 1: find kettle]"));
        assert!(view.contains("Note Buffered: [kettle found at stoveburner 3]"));
    }
}
