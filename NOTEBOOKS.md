# FT09 notebooks

| Notebook | Purpose | Output |
| --- | --- | --- |
| `01_prompt_experiments.ipynb` | Test prompts with LISA or another compatible API | `experiments/` |
| `02_data_collection.ipynb` | Record human/LLM-assisted attempts and export the dataset | `dataset/` |
| `03_baseline_evaluation.ipynb` | Evaluate the local VLM before fine-tuning | `evaluation/` |

Shared functions are in `ft09_shared.py`, model connections in `ft09_models.py`, and recording/Parquet code in `data_pipeline.py`. The original notebook keeps the earlier saved runs.

Run the Dependencies cell first. Restart the kernel if packages were installed or upgraded, then continue below it. Local VLM runs require a GPU. After editing shared Python files, restart the kernel.

Start evaluation with the three-turn connection check. For the full baseline, choose the final model, levels 1–5, five attempts per level and `MAX_STEPS = None`. Keep settings fixed for the before/after comparison. Level 6 stays separate and out of training.

Use unused attempt IDs for new collection runs. Merge and ZIP export are at the end of the collection notebook; raw model runs and evaluation results are excluded. Evaluation folders contain the configuration, turn records, request logs and result summaries.
