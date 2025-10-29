# Environment

This folder tracks Python dependency specifications for the Colab workflow.

- `requirements.txt` is installed inside Colab via `pip install -r env/requirements.txt`.
- `environment.yml` offers a conda-style alternative for local reproducibility.
- `pip_freeze_reference.txt` is a placeholder for recording an exact freeze after validation.

Update the files whenever dependencies change to keep notebooks reproducible.
