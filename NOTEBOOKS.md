# FT09 notebooks

| Notebook | Purpose | Output |
| --- | --- | --- |
| `01_prompt_experiments.ipynb` | Prompt experiments through a compatible API | `experiments/` |
| `02_data_collection.ipynb` | Human/LLM-assisted recording, dataset merge and ZIP export | `dataset/` |
| `03_baseline_evaluation.ipynb` | Local Unsloth VLM evaluation before fine-tuning | `eval/runs/` |

Shared functions: `ft09_shared.py`. Model connections: `ft09_models.py`. Recording/Parquet: `data_pipeline.py`. The original notebook keeps earlier runs.

Install dependencies using the notebook cell, then restart the kernel. Evaluation requires a university CUDA GPU. Default: Qwen3.5-4B, BF16; precision is configurable. Start with the Level 1 three-turn check in two sessions. Once verified, freeze the setup/tag `eval-v1`, set `PHASE = "baseline"` and select assigned levels. Official batches use five attempts at temperature 1.0 and full per-level limits. Level 6 is separate.

Keep the same run name to resume. Finished attempts are skipped; partial attempts replay saved clicks and recover memory. Settings, code hashes and package versions are checked. Results include the saved prompt, config/environment, raw replies, Parquet records and CSV summaries. Keep the complete folder.

After changing shared files, restart the kernel. Use unused attempt IDs for data collection. Evaluation and raw model logs are excluded from the training export.
