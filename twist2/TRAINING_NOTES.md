# Training Notes for TWIST2 Docker

## Quick Training Test

The Docker environment comes pre-configured to work with the example motions for quick testing:

```bash
# Inside container
cd /workspace/twist2
bash train.sh test_experiment cuda:0
```

This will train using the 10 example motion files included in `assets/example_motions/`.

---

## WandB (Weights & Biases) Configuration

WandB is **disabled by default** in the Docker environment to avoid login prompts and connection issues.

### Current Setting
- `WANDB_MODE=disabled`
- `WANDB_DISABLED=true`

### To Enable WandB (Optional)

If you want to use WandB for experiment tracking:

1. Create a WandB account at https://wandb.ai
2. Get your API key from https://wandb.ai/authorize
3. Inside the container, run:
```bash
wandb login YOUR_API_KEY
export WANDB_MODE=online
export WANDB_DISABLED=false
```

---

## Using Full TWIST2 Dataset

The example motions are great for testing, but for real training you'll want the full dataset.

### Steps:

1. **Download the full TWIST2 dataset** from the Google Drive link in the main repository README

2. **Place it on your host machine**, for example:
```bash
/home/youruser/datasets/TWIST2_full/
```

3. **Mount it in docker-compose.yml**:
```yaml
volumes:
  - ./isaacgym:/opt/isaacgym
  - ./twist2:/workspace/twist2
  - /home/youruser/datasets/TWIST2_full:/workspace/datasets/TWIST2_full:ro
```

4. **Update the motion config**:
```bash
# Inside container
cd /workspace/twist2/legged_gym/motion_data_configs
vim twist2_dataset.yaml
```

Change line 2 to:
```yaml
root_path: /workspace/datasets/TWIST2_full
```

5. **Update training config to use full dataset**:
```bash
cd /workspace/twist2/legged_gym/legged_gym/envs/g1
vim g1_mimic_future_config.py
```

Change line 75 to:
```python
motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/twist2_dataset.yaml"
```

---

## Training Configuration

### Quick Test (Example Motions)
- **Dataset**: 10 example walking motions
- **Training time**: Fast (good for testing setup)
- **Use case**: Verify environment, quick experiments

### Full Training (Full Dataset)
- **Dataset**: Thousands of motion files
- **Training time**: Hours to days
- **Use case**: Research, production models

---

## Common Issues

### "No motion files found"
- **Cause**: Motion paths don't match actual files
- **Fix**: Verify `root_path` in YAML config points to correct directory

### "WandB login prompt"
- **Cause**: Environment variables not set
- **Fix**: Rebuild container with updated Dockerfile, or run:
```bash
export WANDB_MODE=disabled
export WANDB_DISABLED=true
```

### "CUDA out of memory"
- **Cause**: Too many parallel environments
- **Fix**: Reduce `num_envs` in task config or use smaller batch size

---

## Training Commands Reference

```bash
# Basic training
bash train.sh experiment_name cuda:0

# Training with custom config
bash train.sh exp_name cuda:0 --num_envs 2048

# Resume training
bash train.sh exp_name cuda:0 --resume

# Training in headless mode (no visualization)
bash train.sh exp_name cuda:0 --headless
```

---

## Output Locations

Training outputs are saved to:
```
twist2/legged_gym/legged_gym/logs/g1_stu_future/YOUR_EXPERIMENT_NAME/
```

This includes:
- `model_*.pt` - PyTorch checkpoints
- `config.yaml` - Training configuration
- Training logs and metrics

---

## Next Steps

After successful training:
1. Export model to ONNX: `bash to_onnx.sh path/to/model_*.pt`
2. Test in simulation: `bash sim2sim.sh`
3. Deploy to real robot: `bash sim2real.sh`

See `VERIFICATION_GUIDE.md` for complete testing instructions.

