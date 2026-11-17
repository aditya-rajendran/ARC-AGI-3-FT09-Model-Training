# FT09 prompt pilot

All runs in this report started from a fresh Level 1 board with a limit of 15 turns. LISA Pro received the current screenshot and, from turn 2 onward, the previous click, the number of cells that changed, and whether the level was solved. Coordinates were always on the 64 × 64 game grid. None of these runs solved Level 1.

## Try 1 · A short reason before each click

The first prompt asked LISA to point out something it could see on the board or explain what it wanted to test, then give one click on the next line. The screenshot was 128 × 128 (scale 2).

System prompt for this run

```text
You are playing FT09. Solve the current level by choosing one click per turn.

Use the current board and the reported effect of the previous click.
In one or two sentences, name a visible detail or an observed click effect
and explain why it supports your next move. If you are unsure of a rule,
say what this click will test and what you expect to change.

ACTION6 x y clicks one position on a 64 × 64 grid. Use integer coordinates
from 0 to 63. The origin is at the top left. x increases to the right and y
increases downward. Use these coordinates even if the image is enlarged.

Reply with exactly two lines:
REASONING: <brief observation and reason for the click>
ACTION: ACTION6 x y
```

All 15 replies could be parsed, but only one click changed the frame. LISA clicked the centre marker at `(48, 48)` three times, with no change each time. On turn 11, it clicked `(48, 38)` expecting the colour to be copied to the other boards. One tile in the active puzzle changed colour instead. The explanation made it possible to check the prediction against what happened.

## Try 2 · Ask for a different click after no change

The prompt now told LISA to try another position after a click that left the board unchanged. The exact wording for tries 2–5 was not saved in the report. The notes below describe the recorded changes. It still received only the previous click's outcome. The result was the same as in the first run. One of 15 clicks changed the frame, and LISA returned to a coordinate it already knew was ineffective on the same board twice.

## Try 3 · Show which clicks did nothing

The next request included a list of coordinates that had left the current board unchanged. The list cleared whenever the frame changed. LISA made 15 clicks, four of which changed the frame. It avoided every coordinate on the ineffective-click list in that run.

Two of the four changes came from clicking `(47, 41)` twice in a row. The second click changed the tile back to its previous colour. A frame change therefore shows that the game reacted, while progress has to be checked on the board.

## Try 4 · Send a larger screenshot

The screenshot sent to LISA increased from 128 × 128 to 512 × 512 (scale 8). The click coordinates still ranged from 0 to 63. Six of 13 clicks changed the frame. The request for turn 14 timed out before a click was applied.

On turn 6, LISA recognised the unframed boards as examples and clicked an editable tile in the framed puzzle. It went back to clicking example boards on turns 9 and 11. It also read a count of 38 changed cells as evidence of a rotation or a wider change. That count can come from a single tile changing colour and two cells in the move bar updating.

## Try 5 · Carry a short finding to the next turn

LISA could now leave a short note for the next request. The idea came from the fine-tuning lecture (slides 38 and 41), where each turn starts with a fresh request and a finding or hypothesis can be passed forward. The note was sent alongside the previous click's outcome.

Four of five clicks changed the frame before the request for turn 6 timed out. The note helped carry forward tile coordinates and the observation that a tile could change colour. It also carried forward the mistaken idea that the active board should match the top-right example. A note can keep an idea in play for several turns, so it matters whether LISA describes it as something it observed or something it still needs to test.

## Try 6 · Keep memory and drop the ineffective-click list

In the current notebook, `MEMORY` is an optional line after `REASONING` and `ACTION`. The request still reports the last click and its outcome, including zero changed cells, and passes along the saved note when there is one. The longer list of ineffective coordinates was removed.

System prompt in the current notebook

```text
You are playing FT09. Your goal is to solve the current level.
The rules are hidden. Work them out from the board and the effects of your clicks.

Use the current screenshot and any click feedback provided. If a click left
the current board unchanged, try a different position. If a click revealed
something about how the game works, use that observation to choose your next move.
When you are unsure, choose a click that tests a specific idea and say what
you expect to change.

Choose exactly one click per turn. ACTION6 x y clicks a position on a 64 × 64
grid. Use integers from 0 to 63. The origin is the top-left corner, x increases
to the right, and y increases downward. Use these grid coordinates even when
the screenshot is enlarged.

Write each field on its own line, in this exact order:
REASONING: <short explanation>
ACTION: ACTION6 x y
MEMORY: <optional finding or hypothesis>

Always put ACTION on line 2. If there is nothing new to remember, omit
the MEMORY line. Use plain text with no bullets or code fences.
```

## Try 7 · Rerun with the same prompt

This run kept the scale-8 screenshot, the previous click’s outcome, and the optional memory note. The ineffective-click list was still off. The previous run timed out after one click. This time all 15 replies passed the parser. Nine clicks changed the frame and six left it unchanged. Level 1 was still unsolved at the turn limit.

The first click on an example board did nothing. The second click, inside the framed puzzle, changed 38 grid cells. By turn 5, LISA connected that count to a single tile changing colour. Some of the nine changes were tests of the colour cycle. On turns 3 and 4, for instance, it clicked the same tile twice, toggling it back.

LISA worked out where it could change tiles, but kept changing its mind about the goal. It tried matching the bottom-left example, then switched to a majority-colour rule on turn 13. Neither approach solved the puzzle. It treated the centre marker as something to click and missed that the marker gives clues about the surrounding tiles. The final centre click left the board unchanged.

No exact click coordinate that had already returned zero changes was repeated in this run. LISA still tried several different spots on the reference boards and two positions on the centre marker. One completed run gives us an example of what happens without the list, but is too little to judge whether it helps overall.

