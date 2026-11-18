"""FT09 step log and Parquet export.

One record per model turn. The system prompt and the text-grid function stay
in the notebook; this module runs the attempt loop and writes the rows.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Decisions from 16 Nov 2026: five attempts, these temperatures, Lisa-Pro rows.
ATTEMPT_TEMPERATURES = (0.2, 0.4, 0.6, 0.8, 1.0)
MAX_PARSE_RETRIES = 2
SOURCE = "lisa-pro"
RENDER_SCALE = 8

# Appended only on a retry. It restates the reply format and does not add game rules.
FORMAT_REMINDER = (
    "Your previous reply could not be parsed or the request timed out. "
    "Reply again in plain text, with one field per line, in this exact order:\n"
    "REASONING: <brief explanation of why you chose this click>\n"
    "ACTION: ACTION6 x y\n"
    "MEMORY: <short summary of useful findings, failed tests, and open hypotheses>\n"
    "Omit MEMORY if there is nothing useful to update. "
    "Use no bullets, blank lines, or code fences."
)

SCHEMA = pa.schema(
    [
        ("level", pa.int16()),
        ("attempt", pa.int16()),
        ("step", pa.int16()),
        ("temperature", pa.float32()),
        ("image_before", pa.binary()),
        ("grid_before", pa.list_(pa.list_(pa.int16()))),
        ("grid_after", pa.list_(pa.list_(pa.int16()))),
        ("grid_text", pa.string()),
        ("history", pa.list_(pa.string())),
        ("reasoning", pa.string()),
        ("memory", pa.string()),
        ("action", pa.string()),
        ("valid", pa.bool_()),
        ("retries", pa.int16()),
        ("changed_cells", pa.int32()),
        ("state", pa.string()),
        ("level_solved", pa.bool_()),
        ("source", pa.string()),
        ("completion_tokens", pa.int32()),
    ]
)


def step_limit(level_number: int, baseline_actions: list[int]) -> int:
    """Return baseline_actions × 1.5, rounded up. Levels are 1-based."""
    if not 1 <= level_number <= len(baseline_actions):
        raise ValueError(
            f"Level {level_number} is outside baseline_actions "
            f"(1–{len(baseline_actions)})."
        )
    return math.ceil(baseline_actions[level_number - 1] * 1.5)


def load_baseline_actions(environments_dir: Path | str, game_id: str = "ft09") -> list[int]:
    """Read baseline_actions from the local FT09 metadata."""
    root = Path(environments_dir)
    matches: list[tuple[float, list[int]]] = []
    for path in root.rglob("metadata.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_id = str(meta.get("game_id", ""))
        base_id = raw_id.split("-", 1)[0]
        actions = meta.get("baseline_actions")
        if base_id == game_id and isinstance(actions, list) and actions:
            matches.append((path.stat().st_mtime, [int(value) for value in actions]))
    if not matches:
        raise FileNotFoundError(
            f"No {game_id} metadata with baseline_actions under {root}. "
            "Run the notebook from the project folder that contains environment_files/."
        )
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def image_to_png_bytes(image) -> bytes:
    """PNG bytes of the board shown to the model, before the click."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def grid_to_lists(frame) -> list[list[int]]:
    """64×64 frame as nested ints so Parquet can store the raw grid."""
    return [[int(value) for value in row] for row in np.asarray(frame)]


def make_record(
    *,
    level: int,
    attempt: int,
    step: int,
    temperature: float,
    image,
    grid_before,
    grid_after,
    grid_text: str,
    history: list[str],
    reasoning: str,
    memory: str,
    action: str,
    valid: bool,
    retries: int,
    changed_cells: int,
    state: str,
    level_solved: bool,
    completion_tokens: int,
    source: str = SOURCE,
) -> dict:
    """Build one step record. `memory` is the note that will be sent on the next turn."""
    return {
        "level": int(level),
        "attempt": int(attempt),
        "step": int(step),
        "temperature": float(temperature),
        "image_before": image_to_png_bytes(image),
        "grid_before": grid_to_lists(grid_before),
        "grid_after": grid_to_lists(grid_after),
        "grid_text": grid_text or "",
        "history": [str(item) for item in history],
        "reasoning": reasoning or "",
        "memory": memory or "",
        "action": action or "",
        "valid": bool(valid),
        "retries": int(retries),
        "changed_cells": int(changed_cells),
        "state": str(state),
        "level_solved": bool(level_solved),
        "source": source,
        "completion_tokens": int(completion_tokens),
    }


# Shared sink for log_step(record). Attempt runs pass their own list instead.
STEP_LOG: list[dict] = []


def log_step(record: dict, records: list[dict] | None = None) -> None:
    """Append one step. log_step(record) writes to STEP_LOG."""
    target = STEP_LOG if records is None else records
    target.append(record)


def write_parquet(records: list[dict], path: Path | str) -> Path:
    """Write step records to one Parquet file. An empty list still writes the schema."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(records), schema=SCHEMA)
    pq.write_table(table, out)
    return out


def export_training_dataset(dataset_dir: Path | str, path: Path | str | None = None) -> Path:
    """Merge levels 1–5. Level 6 is held out and is not read."""
    folder = Path(dataset_dir)
    tables = []
    missing = []
    for level in range(1, 6):
        file = folder / f"ft09_level_{level}.parquet"
        if not file.exists():
            missing.append(level)
            continue
        table = pq.read_table(file, schema=SCHEMA)
        found = set(table.column("level").to_pylist())
        if found != {level}:
            raise ValueError(
                f"{file.name} should contain only level {level}, found {sorted(found)}."
            )
        tables.append(table)
    if missing:
        raise FileNotFoundError(
            f"Missing Parquet for levels {missing}. "
            "Finish those level runs before exporting the training set."
        )
    merged = pa.concat_tables(tables)
    out = Path(path) if path is not None else folder / "ft09_train_levels_1_5.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, out)
    return out


def _state_name(obs) -> str:
    return str(getattr(obs.state, "name", obs.state))


def _copy_frame(obs):
    return np.array(obs.frame[-1], copy=True)


def run_attempt(
    *,
    reset,
    render,
    step,
    done,
    grid_text_fn,
    ask,
    prompt_for,
    level_number: int,
    attempt: int,
    temperature: float,
    max_steps: int,
    records: list[dict],
    on_step=None,
    source: str = SOURCE,
) -> list[str]:
    """Play one attempt and append one record per turn.

    `ask` returns `(move, completion_tokens)`. `move` is
    `(reasoning, x, y, memory)` or None. A timeout is reported by raising
    TimeoutError; it uses one try. After two failed retries the turn is stored
    with valid=False and the attempt continues.

    The memory note starts empty and is replaced only when the model sends a
    new one. The same note is passed into `ask` on the next turn.
    """
    obs = reset()
    if obs is None:
        raise RuntimeError("FT09 reset returned no observation.")

    history: list[str] = []
    memory_note = ""
    initial_completed = int(obs.levels_completed)
    applied: list[str] = []

    for step_index in range(1, max_steps + 1):
        image = render()
        grid_before = _copy_frame(obs)
        history_before = list(history)
        prompt = prompt_for(obs, history_before)
        text_grid = grid_text_fn(grid_before)

        move = None
        tokens_spent = 0
        retries = 0
        for try_index in range(MAX_PARSE_RETRIES + 1):
            reminder = "" if try_index == 0 else FORMAT_REMINDER
            try:
                parsed, tokens = ask(
                    image,
                    text_grid,
                    history_before,
                    prompt,
                    memory_note,
                    temperature,
                    reminder,
                )
            except TimeoutError:
                parsed, tokens = None, 0
            tokens_spent += int(tokens or 0)
            if parsed is not None:
                move = parsed
                break
            if try_index < MAX_PARSE_RETRIES:
                retries += 1

        if move is None:
            record = make_record(
                level=level_number,
                attempt=attempt,
                step=step_index,
                temperature=temperature,
                image=image,
                grid_before=grid_before,
                grid_after=grid_before,
                grid_text=text_grid,
                history=history_before,
                reasoning="",
                memory=memory_note,
                action="",
                valid=False,
                retries=retries,
                changed_cells=0,
                state=_state_name(obs),
                level_solved=False,
                completion_tokens=tokens_spent,
                source=source,
            )
            log_step(record, records)
            if on_step is not None:
                on_step(
                    {
                        "record": record,
                        "history": history_before,
                        "image": image,
                        "x": None,
                        "y": None,
                        "new_note": "",
                    }
                )
            continue

        reasoning, x, y, new_note = move
        x = int(x)
        y = int(y)
        action = f"ACTION6 {x} {y}"
        new_note = (new_note or "").strip()
        obs = step(x, y)
        if obs is None:
            raise RuntimeError(f"FT09 step failed for {action}.")
        grid_after = _copy_frame(obs)
        changed_cells = int(np.count_nonzero(grid_before != grid_after))
        if new_note:
            memory_note = new_note
        level_solved = int(obs.levels_completed) > initial_completed
        history.append(
            f"{action} | changed cells: {changed_cells} "
            f"| level solved: {'yes' if level_solved else 'no'}"
        )
        record = make_record(
            level=level_number,
            attempt=attempt,
            step=step_index,
            temperature=temperature,
            image=image,
            grid_before=grid_before,
            grid_after=grid_after,
            grid_text=text_grid,
            history=history_before,
            reasoning=reasoning,
            memory=memory_note,
            action=action,
            valid=True,
            retries=retries,
            changed_cells=changed_cells,
            state=_state_name(obs),
            level_solved=level_solved,
            completion_tokens=tokens_spent,
            source=source,
        )
        log_step(record, records)
        applied.append(history[-1])
        if on_step is not None:
            on_step(
                {
                    "record": record,
                    "history": history_before,
                    "image": image,
                    "x": x,
                    "y": y,
                    "new_note": new_note,
                }
            )
        if level_solved or done():
            break

    return applied
