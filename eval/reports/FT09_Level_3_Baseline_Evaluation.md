# FT09 Level 3 Baseline Evaluation

Qwen3.5-4B solved **0/5 attempts within 35 decision turns per attempt**. It sometimes found interactive tiles, but failed to turn the observed changes into a consistent understanding of the game. Formatting errors and truncation also affected performance.

## Evaluation setup

This evaluation measures the pretrained model before fine-tuning on the FT09 dataset. Each attempt started with a fresh Level 3 board and empty model memory. The rules were hidden.

| Setting | Value |
| --- | --- |
| Model | `unsloth/Qwen3.5-4B` |
| Model revision | `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636` |
| Precision | BF16 |
| GPU | NVIDIA RTX 4000 Ada Generation |
| Temperature | 1.0 |
| Top p and top k | 0.8 and 20 |
| Maximum generated tokens per reply | 1,024 |
| Maximum context length configured | 8,192 tokens |
| Extended thinking | Disabled |
| Screenshot | 512 × 512 pixels, rendered from the 64 × 64 board |
| Input each turn | Current screenshot, text grid, previous-click feedback and saved model memory |
| Attempts | 5 |
| Decision-turn limit per attempt | 35 |
| Retries after an invalid reply | Up to 2 |

A turn counts towards the limit even if all three requests fail. An accepted reply contains a parsable `ACTION6 x y` with coordinates in range; this does not establish that its reasoning or move is correct.

## Results

| Attempt | Solved | Executed clicks | Clicks with a visible change | Unchanged clicks | Turns without a valid action | Generated tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | No | 33 | 12 | 21 | 2 | 21,507 |
| 2 | No | 32 | 2 | 30 | 3 | 25,557 |
| 3 | No | 31 | 4 | 27 | 4 | 27,606 |
| 4 | No | 31 | 6 | 25 | 4 | 21,558 |
| 5 | No | 35 | 6 | 29 | 0 | 15,028 |
| **Total** | **0/5** | **162** | **30** | **132** | **13** | **111,256** |

All attempts reached the turn limit, with no recorded timeouts or technical failures. **81.5% of executed clicks changed nothing.** Visible changes do not necessarily indicate progress. Clicks needed to solve the level are N/A because no attempt succeeded.

Token totals include rejected replies and retries. Summary counts agree with the Parquet records and request log; changed-cell counts match the saved before and after arrays.

## Game reasoning and use of feedback

**The move bar was mistaken for the goal.** The bar shows remaining moves, but the model treated it as a row to fill. In attempt 4, turns 9–16, it clicked `(62, 63)` eight consecutive times, expecting the bottom row to become complete. Every click changed zero cells. Acknowledging the failed clicks did not lead it to reject the hypothesis.

**Incorrect memory overrode accurate feedback.** In attempt 1, turn 10, clicking `(56, 16)` changed zero cells. The next user message correctly reported this, but the model claimed in turn 11 that the click had caused 37 changes. That result belonged to `(40, 16)` in turn 9. The model saved the incorrect association in memory and continued reasoning from it.

**Repeated changes were not recognised as a colour toggle.** In attempt 5, turns 9–13, the model clicked `(40, 14)` five times. The tile alternated between red and orange. After the first two clicks, the puzzle area had returned to its previous appearance while some move budget had been spent. The explanations still suggested that clicking the tile might clear the pattern or solve the level, rather than identifying the toggle and using it to satisfy the clues.

The model also interpreted 36–37 changed cells as a large interaction or chain reaction. The arrays show 36 changed pixels in one tile, sometimes with one additional pixel changing in the move bar. These counts refer to pixels in the native 64 × 64 observation, not separate tiles.

## Response formatting and truncation

The five attempts produced 280 requests, including retries:

| Request outcome | Count |
| --- | ---: |
| Accepted reply | 162 |
| Rejected reply marked as truncated | 59 |
| Other rejected reply | 59 |
| **Total** | **280** |

Replies were rejected for missing actions, incorrect field order or extra lines. For example, attempt 1, turn 14, request 1 put `MEMORY` before `ACTION`; the final request in turn 33 added a separate sentence after its memory line.

Retries recovered many failures: 118 rejected replies resulted in 13 turns without an executed action. Truncated replies were rejected even if part of an action was present. These failures must be considered alongside the model’s game reasoning.

## Interpretation and later comparison

The model did not develop a reliable interpretation of the clues under this configuration. Its failure involved both game understanding and response compliance; 0/5 does not establish that it could never solve Level 3 with a different setup or larger budget.

After fine-tuning, repeat the same configuration and compare success, unchanged clicks, formatting failures, recognition of colour toggling and use of contradictory feedback.

## Saved evidence

Run folder relative to the project root:

```text
eval/runs/unsloth--Qwen3.5-4B/bf16/bkatsiaryna/baseline_01/
```

The review uses Level 3 attempt files 001–005 and their summaries, Level 3 entries in `requests.jsonl`, and the saved `config.json` and `system_prompt.txt`. Game behaviour was checked against the local `ft09.py` source and saved board states. The source was used for review, not supplied as rules to the evaluated model.
