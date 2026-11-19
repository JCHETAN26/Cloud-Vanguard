# training/ — Milestone 3 (not started)

This directory is a placeholder that fixes where the ML pipeline lives in the monorepo. **There is no training
code yet**; nothing here has been run or benchmarked.

Planned scope (Milestone 3):

| Area | Plan |
|---|---|
| Fine-tuning | PyTorch distributed (DDP/FSDP) PEFT/LoRA adapters |
| Quantization | post-training quantization of the merged or base model for serving |
| Data | Apache Arrow / Parquet datasets streamed into the training loop |
| Tracking | MLflow runs, params, metrics and artifacts |
| Evaluation | vLLM-served candidates evaluated through the CloudVanguard gateway (which already traces every request via OpenTelemetry) |

Layout reserved for it:

```
training/
  pyproject.toml                     package metadata; heavy deps are optional extras
  src/cloudvanguard_training/        package (empty until Milestone 3)
  configs/                           (planned) experiment configs
  tests/                             (planned) unit tests runnable on CPU
```

`pyproject.toml` declares the dependency groups but nothing installs them today, so the repo's existing
C++/Go/Python checks do not depend on PyTorch.
