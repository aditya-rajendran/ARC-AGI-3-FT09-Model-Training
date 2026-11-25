"""Shared FT09 game, prompt, recording and evaluation helpers.

Importing this module does not start a game, load a model or read API keys.
The original notebook remains a historical record of earlier runs.
"""
from __future__ import annotations

from contextlib import chdir
from copy import copy
from collections import Counter, defaultdict
from pathlib import Path
from io import BytesIO
import csv
import json
import math
import os
import re
import tempfile
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from data_pipeline import (
    SCHEMA, RENDER_SCALE, image_to_png_bytes, load_baseline_actions,
    make_record, run_attempt, step_limit, write_parquet,
)

PROJECT_DIR = Path(__file__).resolve().parent
ENVIRONMENTS_DIR = PROJECT_DIR / "environment_files"
DATASET_DIR = PROJECT_DIR / "dataset"
RECORDINGS_DIR = PROJECT_DIR / "recordings"
_arcade = None
_baseline_actions = None


def configure_project(project_dir=None):
    """Set paths explicitly when the notebook is launched outside the project."""
    global PROJECT_DIR, ENVIRONMENTS_DIR, DATASET_DIR, RECORDINGS_DIR
    global _arcade, _baseline_actions
    root = Path(project_dir or Path(__file__).resolve().parent).expanduser().resolve()
    if not (root / "environment_files").is_dir():
        raise FileNotFoundError(f"environment_files/ was not found under {root}")
    if root != PROJECT_DIR:
        _arcade = None
        _baseline_actions = None
    PROJECT_DIR = root
    ENVIRONMENTS_DIR = root / "environment_files"
    DATASET_DIR = root / "dataset"
    RECORDINGS_DIR = root / "recordings"
    return root


def get_baseline_actions():
    global _baseline_actions
    if _baseline_actions is None:
        _baseline_actions = load_baseline_actions(ENVIRONMENTS_DIR)
    return list(_baseline_actions)


def get_arcade():
    global _arcade
    if _arcade is None:
        import arc_agi
        from arc_agi import OperationMode
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        with chdir(PROJECT_DIR):
            _arcade = arc_agi.Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=str(ENVIRONMENTS_DIR),
                recordings_dir=str(RECORDINGS_DIR),
            )
    return _arcade

GAME_PROMPT = 'You are playing FT09. Your goal is to solve the current level.\nThe rules are hidden. Work them out from the board and the effects of your clicks.\n\nUse the current screenshot, the text grid, the previous click\'s outcome, and\nany saved memory.\nWhen you are unsure, choose a click that tests a specific idea. Say what visible\nchange you expect and where, and save that prediction in MEMORY.\n\nEach turn also includes a text grid of the same board. It has 31 lines of\n32 hexadecimal characters. The character in column c of line r is the colour\nindex at grid position x = 2 × c, y = 2 × r (both counted from 0). The last\nline, "bar: n/64", counts the cells of colour 12 in the bottom row.\n\nOn the next turn, check the predicted spot in the screenshot. The changed-cell\ncount tells you how much changed. Use the screenshot to check what changed and\nwhere.  If the expected change did not happen, record the failed test in MEMORY\nand try a different test. If you expected the click to solve the level and it\ndid not, reconsider the rule behind that prediction.\n\nChoose exactly one click per turn. ACTION6 x y clicks a position on the 64 × 64\ngame grid. Use integers from 0 to 63, even when the screenshot is enlarged.\nThe origin is at the top left. x increases to the right and y downward.\n\nKeep earlier findings that still matter. Label untested ideas as hypotheses.\nOmit MEMORY if there is nothing useful to update.\nUse no bullets, blank lines, or code fences.\n\nReply in plain text, with one field per line, in this exact order:\nREASONING: <brief explanation of why you chose this click>\nACTION: ACTION6 x y\nMEMORY: <short summary of useful findings, failed tests, and open hypotheses>\n'

def parse_answer(text):
    match = re.fullmatch(
        r"REASONING:[ \t]*([^\r\n]+)\r?\n"
        r"ACTION:[ \t]*ACTION6[ \t]+([0-9]{1,2})[ \t]+([0-9]{1,2})[ \t]*"
        r"(?:\r?\nMEMORY:[ \t]*([^\r\n]+))?",
        (text or "").strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    reasoning = match.group(1).strip()
    x, y = map(int, match.group(2, 3))
    if not reasoning or not (0 <= x < 64 and 0 <= y < 64):
        return None

    return reasoning, x, y, (match.group(4) or "").strip()

def frame_to_image(observation, scale=8):
    """Render the newest frame of an observation as a PIL image."""
    from arc_agi.rendering import frame_to_rgb_array
    pixels = frame_to_rgb_array(
        steps=0,
        frame=observation.frame[-1],
        scale=scale,
    )
    return Image.fromarray(pixels)

def grid_to_text(frame):
    """64x64 FT09 frame -> compact text: 31 rows x 32 hex chars (1 char = 2x2 pixels) + bar."""
    board = frame[:62:2, ::2]  # rows 0-61
    bar = int((frame[63] == 12).sum())
    lines = ["".join(f"{v:x}" for v in row) for row in board]
    lines.append(f"bar: {bar}/64")
    return "\n".join(lines)

class FT09Game:
    """Convenience wrapper around the FT09 environment."""

    def __init__(self, arcade, game_id="ft09"):
        self.env = arcade.make(game_id)
        if self.env is None:
            raise RuntimeError(
                "FT09 could not be loaded. Check the local environment files "
                "and the ARC SDK messages."
            )
        self.actions = {a.name: a for a in self.env.action_space}
        self.obs = None

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def step(self, x, y):
        self.obs = self.env.step(
            self.actions["ACTION6"],
            data={"x": x, "y": y},
        )
        return self.obs

    def render(self, scale=8):
        return frame_to_image(self.obs, scale=scale)

    def done(self):
        return self.obs.state.name in {"WIN", "GAME_OVER"}

class FT09SingleLevel(FT09Game):
    def __init__(self, level_number):
        if not isinstance(level_number, int) or not 1 <= level_number <= 6:
            raise ValueError("Wähle ein Level von 1 bis 6.")
        # Lokale Metadaten können Pfade relativ zum Umgebungsordner enthalten.
        with chdir(ENVIRONMENTS_DIR.parent):
            super().__init__(get_arcade())
        self.level_number = level_number

    def reset(self):
        start = super().reset()
        # Der lokale SDK-Wrapper stellt die Spielinstanz über _game bereit.
        engine = self.env._game
        engine.set_level(self.level_number - 1)
        self.obs = copy(start)
        self.obs.frame = [engine.camera.render(engine.current_level.get_sprites())]
        return self.obs

def record_human_attempt(level, attempt, max_steps=None):
    import ipywidgets as widgets
    from ipyevents import Event
    if level not in range(1, 6):
        raise ValueError(
            "Choose an original training level from 1 to 5."
        )

    output = (
        DATASET_DIR / "human"
        / f"ft09_level_{level}_human_{attempt}.parquet"
    )
    if output.exists():
        raise FileExistsError(
            f"{output.name} already exists. Choose another attempt ID."
        )

    limit = (
        step_limit(level, get_baseline_actions())
        if max_steps is None else max_steps
    )
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("max_steps must be a positive integer.")

    game = FT09SingleLevel(level)
    obs = game.reset()
    initial_completed = obs.levels_completed

    records, history = [], []
    image = game.render(scale=RENDER_SCALE)
    selected = None
    pending_save = False
    finished = False

    board = widgets.Image(
        value=image_to_png_bytes(image), format="png"
    )
    board.layout = widgets.Layout(
        width="512px", max_width="100%"
    )

    title = widgets.Label(
        value=f"Level {level} · Attempt {attempt} · Move 1"
    )
    position = widgets.Label(
        value="Click a square on the board."
    )

    reason = widgets.Textarea(
        placeholder="What do you see, and why this click?",
        layout=widgets.Layout(
            width="512px", max_width="100%", height="85px"
        ),
    )
    memory = widgets.Textarea(
        placeholder=(
            "Known rules, useful findings and open hypotheses "
            "before this click."
        ),
        layout=widgets.Layout(
            width="512px", max_width="100%", height="100px"
        ),
    )

    apply = widgets.Button(
        description="Apply click",
        button_style="primary",
        disabled=True,
    )
    finish = widgets.Button(description="Finish recording")
    status = widgets.Label(
        value=(
            "Select a position, enter reasoning and memory, "
            "then apply the click."
        )
    )
    events = Event(source=board, watched_events=["click"])

    def update_button(change=None):
        missing_input = (
            selected is None
            or not reason.value.strip()
            or not memory.value.strip()
        )
        apply.disabled = finished or (
            not pending_save and missing_input
        )

    def select_position(event):
        nonlocal selected
        if finished or pending_save:
            return

        px, py = event.get("dataX"), event.get("dataY")
        if px is None or py is None:
            status.value = "Click inside the board image."
            return
        if not (
            0 <= px < image.width and 0 <= py < image.height
        ):
            return

        selected = (
            int(px * 64 / image.width),
            int(py * 64 / image.height),
        )
        position.value = (
            f"Selected click: ACTION6 "
            f"{selected[0]} {selected[1]}"
        )
        update_button()

    def save():
        temporary = output.with_suffix(".tmp")
        write_parquet(records, temporary)
        temporary.replace(output)

    def end_recording(message):
        nonlocal finished
        finished = True
        events.watched_events = []
        apply.disabled = True
        finish.disabled = True
        reason.disabled = True
        memory.disabled = True
        status.value = message

    def apply_click(button):
        nonlocal obs, image, selected, pending_save

        if finished or apply.disabled:
            return

        apply.disabled = True

        if not pending_save:
            x, y = selected
            before = obs.frame[-1].copy()
            obs = game.step(x, y)
            action = f"ACTION6 {x} {y}"
            solved = obs.levels_completed > initial_completed

            record = make_record(
                level=level,
                attempt=attempt,
                step=len(records) + 1,
                temperature=0.0,
                image=image,
                grid_before=before,
                grid_after=obs.frame[-1],
                grid_text=grid_to_text(before),
                history=list(history),
                reasoning=reason.value.strip(),
                memory=memory.value.strip(),
                action=action,
                valid=True,
                retries=0,
                changed_cells=int(
                    np.count_nonzero(before != obs.frame[-1])
                ),
                state=obs.state.name,
                level_solved=solved,
                completion_tokens=0,
                source="human",
            )
            record["temperature"] = None
            records.append(record)
            pending_save = True

        try:
            save()
        except Exception as exc:
            apply.description = "Retry save"
            status.value = (
                f"Move applied. Saving failed: {exc}. "
                "Press Retry save."
            )
            finish.disabled = True
            reason.disabled = True
            memory.disabled = True
            update_button()
            return

        pending_save = False
        apply.description = "Apply click"
        finish.disabled = False
        reason.disabled = False
        memory.disabled = False

        record = records[-1]
        history.append(
            f"{record['action']} | "
            f"changed cells: {record['changed_cells']} | "
            f"level solved: "
            f"{'yes' if record['level_solved'] else 'no'}"
        )

        image = game.render(scale=RENDER_SCALE)
        board.value = image_to_png_bytes(image)
        selected = None
        reason.value = ""
        memory.value = ""

        position.value = "Click a square on the board."
        status.value = (
            f"{history[-1]} · Saved {len(records)} moves."
        )
        title.value = (
            f"Level {level} · Attempt {attempt} · "
            f"Move {len(records) + 1}"
        )

        if record["level_solved"]:
            end_recording(
                f"Level solved. Saved {len(records)} moves "
                f"to {output}"
            )
        elif game.done() or len(records) >= limit:
            end_recording(
                f"Attempt ended. Saved {len(records)} moves "
                f"to {output}"
            )
        else:
            update_button()

    def finish_recording(button):
        end_recording(
            f"Saved {len(records)} moves to {output}"
            if records else "Finished with no moves recorded."
        )

    events.on_dom_event(select_position)
    reason.observe(update_button, names="value")
    memory.observe(update_button, names="value")
    apply.on_click(apply_click)
    finish.on_click(finish_recording)

    return widgets.VBox([
        title,
        board,
        position,
        widgets.Label(value="Reasoning before this click:"),
        reason,
        widgets.Label(value="Memory before this click:"),
        memory,
        widgets.HBox([apply, finish]),
        status,
    ])

def record_llm_attempt(level, attempt, max_steps=None):
    import ipywidgets as widgets
    if level not in range(1, 6):
        raise ValueError("Choose an original training level from 1 to 5.")

    output = DATASET_DIR / "llm_assisted" / f"ft09_level_{level}_llm_{attempt}.parquet"
    if output.exists():
        raise FileExistsError(f"{output.name} already exists. Choose another attempt ID.")

    limit = step_limit(level, get_baseline_actions()) if max_steps is None else max_steps
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("max_steps must be a positive integer.")

    game = FT09SingleLevel(level)
    obs = game.reset()
    initial_completed = obs.levels_completed
    records, history = [], []
    image = game.render(scale=RENDER_SCALE)
    pending_save = False
    finished = False

    board = widgets.Image(value=image_to_png_bytes(image), format="png")
    board.layout = widgets.Layout(width="512px", max_width="100%")
    title = widgets.Label(value=f"LLM-assisted · Level {level} · Attempt {attempt} · Move 1")
    action_input = widgets.Text(
        placeholder="30 24",
        layout=widgets.Layout(width="512px", max_width="100%"),
    )
    reason = widgets.Textarea(
        placeholder="What do you see, and why this click?",
        layout=widgets.Layout(width="512px", max_width="100%", height="85px"),
    )
    memory = widgets.Textarea(
        placeholder="Paste the updated MEMORY for this turn (before the click).",
        layout=widgets.Layout(width="512px", max_width="100%", height="85px"),
    )
    apply = widgets.Button(description="Apply action", button_style="primary", disabled=True)
    finish = widgets.Button(description="Finish recording")
    status = widgets.Label(value="Enter ACTION6 x y, reasoning and memory, then apply the action.")
    def parse_action():
        match = re.fullmatch(
            r"(\d+)[,\s]+(\d+)",
            action_input.value.strip(),
        )
        if not match:
            return None
        x, y = map(int, match.groups())
        return (x, y) if 0 <= x <= 63 and 0 <= y <= 63 else None

    def update_button(change=None):
        missing_input = parse_action() is None or not reason.value.strip() or not memory.value.strip()
        apply.disabled = finished or (not pending_save and missing_input)

    def save():
        temporary = output.with_suffix(".tmp")
        write_parquet(records, temporary)
        temporary.replace(output)

    def end_recording(message):
        nonlocal finished
        finished = True
        action_input.disabled = True
        apply.disabled = True
        finish.disabled = True
        reason.disabled = True
        memory.disabled = True
        status.value = message

    def apply_click(button):
        nonlocal obs, image, pending_save
        if finished or apply.disabled:
            return
        apply.disabled = True
        if not pending_save:
            x, y = parse_action()
            before = obs.frame[-1].copy()
            obs = game.step(x, y)
            action = f"ACTION6 {x} {y}"
            solved = obs.levels_completed > initial_completed
            record = make_record(
                level=level, attempt=attempt, step=len(records) + 1,
                temperature=0.0, image=image,
                grid_before=before, grid_after=obs.frame[-1],
                grid_text=grid_to_text(before), history=list(history),
                reasoning=reason.value.strip(), memory=memory.value.strip(), action=action,
                valid=True, retries=0,
                changed_cells=int(np.count_nonzero(before != obs.frame[-1])),
                state=obs.state.name, level_solved=solved,
                completion_tokens=0, source="llm-assisted",
            )
            record["temperature"] = None
            records.append(record)
            pending_save = True
        try:
            save()
        except Exception as exc:
            apply.description = "Retry save"
            status.value = f"Move applied. Saving failed: {exc}. Press Retry save."
            finish.disabled = True
            reason.disabled = True
            memory.disabled = True
            action_input.disabled = True
            update_button()
            return

        pending_save = False
        apply.description = "Apply action"
        finish.disabled = False
        reason.disabled = False
        memory.disabled = False
        action_input.disabled = False
        record = records[-1]
        history.append(
            f"{record['action']} | changed cells: {record['changed_cells']} "
            f"| level solved: {'yes' if record['level_solved'] else 'no'}"
        )
        image = game.render(scale=RENDER_SCALE)
        board.value = image_to_png_bytes(image)
        reason.value = ""
        memory.value = ""
        action_input.value = ""
        status.value = f"{history[-1]} · Saved {len(records)} moves."
        title.value = f"LLM-assisted · Level {level} · Attempt {attempt} · Move {len(records) + 1}"
        if record["level_solved"]:
            end_recording(f"Level solved. Saved {len(records)} moves to {output}")
        elif game.done() or len(records) >= limit:
            end_recording(f"Attempt ended. Saved {len(records)} moves to {output}")
        else:
            update_button()

    def finish_recording(button):
        end_recording(
            f"Saved {len(records)} moves to {output}" if records else "Finished with no moves recorded."
        )

    action_input.observe(update_button, names="value")
    reason.observe(update_button, names="value")
    memory.observe(update_button, names="value")
    apply.on_click(apply_click)
    finish.on_click(finish_recording)
    return widgets.VBox([
        title, board,
        widgets.Label(value="Action (64 × 64 grid coordinates):"), action_input,
        widgets.Label(value="Reasoning before this click:"), reason,
        widgets.Label(value="Updated memory before this click:"), memory,
        widgets.HBox([apply, finish]), status,
    ])


def new_run_dir(purpose, model_name):
    """Create a unique output directory outside the training collection."""
    if purpose not in {"experiments", "evaluation", "collection_runs"}:
        raise ValueError("Choose experiments, evaluation or collection_runs.")
    model_part = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = PROJECT_DIR / purpose / model_part / stamp
    output.mkdir(parents=True, exist_ok=False)
    return output


def save_run_config(output_dir, backend, prompt=GAME_PROMPT, **settings):
    """Save reproducibility settings. Backend information never includes keys."""
    path = Path(output_dir) / "config.json"
    info = {
        "model": backend.info(), "prompt": prompt,
        "render_scale": RENDER_SCALE,
        "baseline_actions": get_baseline_actions(),
        "max_parse_retries": 2,
        **settings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path


def build_turn_text(text_grid, history, memory_note, format_reminder=""):
    previous = history[-1] if history else "None. This is the first turn."
    text = (
        "The attached image shows the current board.\n"
        f"Text grid of the current board:\n{text_grid}\n"
        f"Previous click: {previous}\nChoose one click."
    )
    if memory_note:
        text += f"\nModel's memory from this run: {memory_note}"
    if format_reminder:
        text += "\n" + format_reminder
    return text


def _save_records_atomic(records, output):
    output = Path(output)
    temporary = output.with_suffix(".tmp")
    write_parquet(records, temporary)
    temporary.replace(output)


def run_level(
    backend, level, output_dir, *, n_attempts=1, temperature=0.7,
    max_steps=None, attempt_start=1, prompt=GAME_PROMPT, show_frames=False,
):
    """Run one level with an explicit backend, saving each attempt separately.

    Existing files are never overwritten. A fresh board and empty memory are
    created by data_pipeline.run_attempt for every attempt. The step limit is a
    decision-turn limit: an invalid turn consumes budget but executes no click.
    """
    if not isinstance(level, int) or not 1 <= level <= 6:
        raise ValueError("Choose a level from 1 to 6.")
    if not isinstance(n_attempts, int) or n_attempts < 1:
        raise ValueError("n_attempts must be a positive integer.")
    if not isinstance(attempt_start, int) or attempt_start < 1:
        raise ValueError("attempt_start must be a positive integer.")
    limit = step_limit(level, get_baseline_actions()) if max_steps is None else max_steps
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("max_steps must be a positive integer or None.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_ids = range(attempt_start, attempt_start + n_attempts)
    outputs = {
        attempt: output_dir / f"ft09_level_{level}_attempt_{attempt:03d}.parquet"
        for attempt in attempt_ids
    }
    for output in outputs.values():
        if output.exists() or output.with_suffix(".summary.json").exists():
            raise FileExistsError(f"Run already exists: {output}. Use a new run directory or attempt_start.")
    game = FT09SingleLevel(level)
    combined = []
    for attempt, output in outputs.items():
        records = []
        backend.set_context(level=level, attempt=attempt)
        request_start = len(backend.request_log)
        started = time.monotonic()
        error = None

        def on_step(event):
            row = event["record"]
            _save_records_atomic(records, output)
            print(
                f"Level {level} | attempt {attempt} | turn {row['step']}/{limit} | "
                f"{row['action'] or 'invalid reply'} | changed {row['changed_cells']} | "
                f"solved {row['level_solved']}", flush=True,
            )
            if row["reasoning"]:
                print(row["reasoning"], flush=True)
            if show_frames:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                extent = (-0.5, 63.5, 63.5, -0.5)
                axes[0].imshow(event["image"], extent=extent, interpolation="nearest")
                if event["x"] is not None:
                    axes[0].scatter(event["x"], event["y"], marker="x", color="lime")
                axes[1].imshow(game.render(), extent=extent, interpolation="nearest")
                axes[0].set_title("Before")
                axes[1].set_title("After")
                fig.tight_layout()
                plt.show()
                plt.close(fig)

        try:
            run_attempt(
                reset=game.reset, render=lambda: game.render(scale=RENDER_SCALE),
                step=game.step, done=game.done, grid_text_fn=grid_to_text,
                ask=backend.ask, prompt_for=lambda state, history: prompt,
                level_number=level, attempt=attempt, temperature=temperature,
                max_steps=limit, records=records, on_step=on_step,
                source="llm-api",
            )
        except (Exception, KeyboardInterrupt) as exc:
            error = exc
            raise
        finally:
            _save_records_atomic(records, output)
            requests = backend.request_log[request_start:]
            solved = any(row["level_solved"] for row in records)
            if error is not None:
                stop_reason = "interrupted" if isinstance(error, KeyboardInterrupt) else "technical_error"
            elif solved:
                stop_reason = "solved"
            elif records and records[-1]["state"] in {"GAME_OVER", "WIN"}:
                stop_reason = "game_ended"
            else:
                stop_reason = "turn_limit"
            summary = {
                "model": backend.model_name, "level": level, "attempt": attempt,
                "solved": solved, "decision_turns": len(records),
                "executed_clicks": sum(row["valid"] for row in records),
                "invalid_turns": sum(not row["valid"] for row in records),
                "invalid_replies": sum(r["status"] == "invalid_reply" for r in requests),
                "timeouts": sum(r["status"] == "timeout" for r in requests),
                "requests": len(requests),
                "generated_tokens_known": sum(r.get("completion_tokens") or 0 for r in requests),
                "token_usage_complete": all(r.get("completion_tokens") is not None for r in requests),
                "temperature": temperature, "turn_limit": limit,
                "stop_reason": stop_reason,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": str(error) if error else "",
                "records_file": output.name,
            }
            output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        combined.extend(records)
    return combined


def summarise_evaluation(output_dir):
    """Write per-attempt and per-level CSVs without discarding failed runs."""
    output_dir = Path(output_dir)
    paths = sorted(output_dir.glob("ft09_level_*_attempt_*.summary.json"))
    if not paths:
        raise FileNotFoundError(f"No attempt summaries under {output_dir}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows.sort(key=lambda r: (r["level"], r["attempt"]))
    if len({(r["model"], r["level"], r["attempt"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate attempt summaries.")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["level"]].append(row)
    aggregate = []
    for level, group in sorted(groups.items()):
        successful = [r for r in group if r["solved"]]
        complete = all(r["token_usage_complete"] for r in group)
        aggregate.append({
            "level": level, "attempts": len(group), "solved": len(successful),
            "success_rate": len(successful) / len(group),
            "average_clicks_when_solved": (
                sum(r["executed_clicks"] for r in successful) / len(successful)
                if successful else None
            ),
            "average_generated_tokens": (
                sum(r["generated_tokens_known"] for r in group) / len(group)
                if complete else None
            ),
            "invalid_replies": sum(r["invalid_replies"] for r in group),
            "timeouts": sum(r["timeouts"] for r in group),
            "technical_failures": sum(r["stop_reason"] in {"technical_error", "interrupted"} for r in group),
        })
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    for row in aggregate:
        print(row)
    return rows, aggregate


def review_steps(path, first_step=1, last_step=5):
    """Display recorded before/after boards and text without modifying a file."""
    import matplotlib.pyplot as plt
    from arc_agi.rendering import frame_to_rgb_array
    rows = pq.read_table(path, schema=SCHEMA).to_pylist()
    for row in rows:
        if not first_step <= row["step"] <= last_step:
            continue
        print(f"Step {row['step']} | {row['action']} | source {row['source']}")
        print("Reasoning:", row["reasoning"])
        print("Memory:", row["memory"])
        print("Changed cells:", row["changed_cells"], "| solved:", row["level_solved"])
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        with Image.open(BytesIO(row["image_before"])) as image:
            axes[0].imshow(image, interpolation="nearest", extent=(-0.5, 63.5, 63.5, -0.5))
        if row["valid"]:
            x, y = map(int, row["action"].split()[1:])
            axes[0].scatter(x, y, marker="x", color="lime")
        after = frame_to_rgb_array(steps=0, frame=np.asarray(row["grid_after"]), scale=4)
        axes[1].imshow(after, interpolation="nearest", extent=(-0.5, 63.5, 63.5, -0.5))
        axes[0].set_title("Before")
        axes[1].set_title("After")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


TRAINING_FOLDERS = (
    "human", "llm_assisted", "llm_generated",
    "llm_generated_exploration", "model_reviewed",
)


def merge_training_files(output=None, *, overwrite=False):
    """Merge the five selected folders. No row filtering or source renaming."""
    output = Path(output or DATASET_DIR / "ft09_train_levels_1_5.parquet").resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; set overwrite=True only to replace the merged output.")
    if any(output.is_relative_to(DATASET_DIR / folder) for folder in TRAINING_FOLDERS):
        raise ValueError("The merged file must be outside the selected source folders.")
    tables, keys = [], set()
    schema = None
    for folder in TRAINING_FOLDERS:
        files = sorted((DATASET_DIR / folder).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files in {DATASET_DIR / folder}")
        for path in files:
            table = pq.read_table(path).replace_schema_metadata(None)
            if schema is None:
                schema = table.schema
            elif not schema.equals(table.schema, check_metadata=False):
                raise ValueError(f"Schema mismatch in {path}; no columns were dropped.")
            for row in table.select(["level", "attempt", "step", "source"]).to_pylist():
                if row["level"] not in range(1, 6):
                    raise ValueError(f"Level outside 1–5 in {path}")
                if row["source"] not in {"human", "llm-assisted", "llm-api", "llm-generated"}:
                    raise ValueError(f"Unexpected source in {path}: {row['source']}")
                key = (row["level"], row["attempt"], row["step"])
                if key in keys:
                    raise ValueError(f"Duplicate record ID {key}; no records were deleted.")
                keys.add(key)
            tables.append(table)
    if {key[0] for key in keys} != {1, 2, 3, 4, 5}:
        raise ValueError("The collection must contain all five training levels.")
    merged = pa.concat_tables(tables).sort_by([
        ("level", "ascending"), ("attempt", "ascending"), ("step", "ascending"),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        pq.write_table(merged, temporary)
        if not pq.read_table(temporary).equals(merged):
            raise RuntimeError("The saved file did not match the merged records.")
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved {len(merged)} rows: {output}")
    return output


def package_dataset(merged_file, readme=None, output=None, *, overwrite=False):
    """Package only the merged Parquet and dataset README, never raw runs."""
    merged_file = Path(merged_file)
    readme = Path(readme or DATASET_DIR / "README.md")
    output = Path(output or DATASET_DIR / "ft09_dataset.zip")
    if not merged_file.is_file() or not readme.is_file():
        raise FileNotFoundError("Both the merged Parquet and README are required.")
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with zipfile.ZipFile(output, mode, zipfile.ZIP_DEFLATED) as archive:
        archive.write(merged_file, arcname=merged_file.name)
        archive.write(readme, arcname="README.md")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("The ZIP did not pass its integrity check.")
    return output
