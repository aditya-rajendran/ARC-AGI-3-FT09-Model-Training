# ARC-AGI-3 FT09 – Base Dataset

Training data for fine-tuning a VLM to play the ARC-AGI-3 game FT09.

## 1. Overview

- Game: FT09 (ARC-AGI SDK 0.9.9, offline environment in `environment_files/`)
- Levels 1–5: training data.
- One row = one move: screenshot before the move, grid before/after, reasoning, memory and action.
- Total: 530 steps (excluding raw model runs).

## 2. How the traces were created

- **human:** Recorded in the notebook recorder. Levels played in order; from Level 2 onward, rules from earlier levels are used as prior knowledge.
- **llm_assisted:** A chat LLM (GPT-6.2 Sol, Gemini 3.5 Flash-Lite, Claude Sonnet) proposed reasoning, action and memory from the screenshot; a human executed the move and pasted the text. Starts without prior knowledge; contains real mistakes and corrections.
- **llm_generated / exploration:** LLM-authored programmatic demonstrations, executed by a deterministic offline replay; explanations composed from board states. Not autonomous model play (see `generation_method` in the file metadata).
- **model_raw:** Autonomous LISA Pro runs via API (temperature 1.0). None solved its level.
- **model_reviewed:** Selected steps from `model_raw`. Actions, images and metadata unchanged; reasoning and memory rewritten to describe only what was observable at that step.

## 3. Review of model runs

| Level | Raw steps | Kept steps | Main problems |
|---|---|---|---|
| 3 | 35 | 7 | Treated markers as rotating "dice", many clicks without effect, repeated toggling of the same tile, 17 invalid steps (output could not be parsed) |
| 4 | 42 | 7 | Hallucinated "vanish" rule, repeated clicks, lost the rule later |
| 5 | 61 | 8 | Wrong goal ("make all green"), colour confusion, 7 invalid steps |
