# Configurations

Centralized YAML files governing datasets, training, slice composition, and attack toggles.

- Edit `data.yaml` with secure dataset paths, ID fields, and chronology assumptions.
- `train_llm.yaml` contains LoRA/QLoRA hyper-parameters and token-budget rules for continual fine-tuning.
- `slices.yaml` tracks temporal windows, artifact locations for member/non-member IDs, and replay variants.
- `attacks.yaml` enables or disables specific membership inference procedures and their hyper-parameters.

Keep configs under version control to enable reproducible experiments and Colab synchronization.
