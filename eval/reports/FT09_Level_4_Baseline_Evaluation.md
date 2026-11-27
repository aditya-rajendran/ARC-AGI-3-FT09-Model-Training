# FT09 Level 4 Baseline Evaluation

Qwen3.5-4B solved **0/5 attempts within 42 decision turns per attempt**. It sometimes found interactive tiles, but failed to turn the observed changes into a consistent understanding of the game. Formatting errors and truncation also affected performance.

## Evaluation setup

This evaluation measures the pretrained model before fine-tuning on the FT09 dataset. Each attempt started with a fresh Level 4 board and empty model memory. The rules were hidden.

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
| Decision-turn limit per attempt | 42 |
| Retries after an invalid reply | Up to 2 |

The turn limit is the agreed baseline action count of 28 multiplied by 1.5. A turn counts towards the limit even if all three requests fail. An accepted reply contains a parsable `ACTION6 x y` with coordinates in range; this does not establish that its reasoning or move is correct.

## Results

| Attempt | Solved | Executed clicks | Clicks with a visible change | Unchanged clicks | Turns without a valid action | Generated tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | No | 40 | 14 | 26 | 2 | 27,872 |
| 2 | No | 38 | 13 | 25 | 4 | 25,583 |
| 3 | No | 36 | 12 | 24 | 6 | 33,016 |
| 4 | No | 40 | 4 | 36 | 2 | 21,414 |
| 5 | No | 40 | 6 | 34 | 2 | 35,061 |
| **Total** | **0/5** | **194** | **49** | **145** | **16** | **142,946** |

All attempts reached the turn limit, with no recorded timeouts or technical failures. **74.7% of executed clicks changed nothing.** The other 49 clicks changed a tile, but that does not establish progress. Clicks needed to solve the level are N/A.

Token totals include rejected replies and retries. Summaries agree with the records and request log. Changed-cell counts, text grids and previous-click feedback match the saved observations.

## Game reasoning and use of feedback

**The three-colour cycle was not recognised.** Level 4 tiles cycle through blue, red and orange, so reaching a chosen colour can require two clicks. In attempt 2, turns 2–4, the model clicked `(38, 38)` three times, changing one tile from blue to red, orange and back to blue. In turns 5–7, it repeated the cycle at `(38, 42)`. Both coordinates fall inside the same tile, with bounds x = 36–41 and y = 38–43.

After those six clicks, the puzzle area matched its appearance before turn 2, while some move budget had been spent. The model described the positions as a red block and a white block and treated the changes as possible pattern completion. It did not use them to establish the cycle or a plan for satisfying the clues.

**The move bar was mistaken for a target.** In attempt 5, turns 9–13, the model clicked `(63, 63)` five consecutive times, expecting to complete the bottom row because the bar showed `63/64`. Across the attempt, it clicked this position 12 times, all without a visible change. The bar actually displays remaining moves.

**Memory confused earlier effects with the latest action.** In attempt 1, turn 11, clicking `(60, 40)` changed zero cells. The next request correctly reported this, but the explanation in turn 12 claimed that the click had caused 37 changes. The model attached an earlier result from its memory to the latest action.

In attempt 5, turns 17–18, it also suggested that the text grid was stale. The logged grids match the saved boards. At turn 18, the actual previous click was `(0, 0)` with zero changes, while memory still discussed the earlier `(32, 15)` click that changed 36 cells.

**Descriptions and click positions did not always match.** In attempt 4, turns 13–17, the model clicked `(8, 10)` five times and described it as colour 9 or colour 2. The saved board shows background, colour index 4; every click changed nothing. In turn 17, it referred to row 10, column 10 of the text grid to justify this action, which does not map to `(8, 10)` under the supplied conversion.

**Changed pixels were mistaken for large puzzle effects.** In attempt 1, turn 6, the model interpreted a previous 37-pixel change as evidence of a reflection operation and a symmetry-based goal. Across all 49 clicks with a visible change, the arrays show 36 changed pixels in one tile, with zero or one additional pixel in the move bar. These are pixels in the native 64 × 64 observation, not separate tiles.

## Response formatting and truncation

The five attempts produced 353 requests, including retries:

| Request outcome | Count |
| --- | ---: |
| Accepted reply | 194 |
| Rejected reply marked as truncated | 74 |
| Other rejected reply | 85 |
| **Total** | **353** |

Among the 85 other rejected replies, 36 placed `MEMORY` before `ACTION`, 33 omitted `ACTION6`, and 16 had other formatting problems. The first request in attempt 1, turn 1, narrated an imagined outcome and proposed another action; these were generated statements, not executed moves.

Overall, 45.0% of requests were rejected. Retries recovered many failures, leaving 16 of 210 turns without an executed click. Truncated replies were rejected even if part of an action was present.

## Interpretation and later comparison

The model sometimes found interactive tiles but did not reliably identify their colour cycle, connect them to clue markers or revise failed hypotheses. The result describes this model, prompt, parser and budget, rather than proving that Level 4 could never be solved with another setup.

After fine-tuning, repeat the same configuration. Compare success, formatting failures, recognition of the three-colour cycle, repeated background/bar clicks, coordinate accuracy and whether actual feedback takes priority over unsupported memory.

## Saved evidence

Run folder relative to the project root:

```text
eval/runs/unsloth--Qwen3.5-4B/bf16/bkatsiaryna/baseline_01/
```

The review uses Level 4 attempt files 001–005 and their summaries, Level 4 entries in `requests.jsonl`, and the saved `config.json` and `system_prompt.txt`. Game behaviour was checked against the local `ft09.py` source and saved board states. The source was used for review, not supplied as rules to the evaluated model.
