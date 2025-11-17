# Notebooks

Colab-ready notebooks implementing the end-to-end continual fine-tuning and membership inference evaluation workflow.

Every notebook now starts with a shared “Persistent Drive + run mode setup” cell. Set `SECURE_LLM_MIA_RUN_MODE` to `subset` (default) or `full` before executing notebook 00 so all downstream notebooks point at the same Drive directories and canonical artifacts.

Run notebooks sequentially (00–12) or orchestrate via `12_run_sweep_driver.ipynb`.
