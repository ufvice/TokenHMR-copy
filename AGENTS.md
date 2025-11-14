# Repository Guidelines

## Project Structure & Module Organization
- `tokenhmr/` is the main PyTorch Lightning project: `train.py`, `eval.py`, `demo.py`, and `lib/` hold the training loop, eval wrappers, Hydra configs, and dataset utilities.
- `tokenization/` contains the VQ-VAE tokenizer training code, configs, and processed pose data; keep downloaded archives beneath `tokenization/tokenization_data/` as described in the README.
- `assets/`, `demo_sample/`, and `fetch_demo_data.sh` support lightweight demos and data acquisition scripts; use them when preparing inputs for `demo.py` and validation runs.
- Keep experiments and checkpoints outside the repo (e.g., `data/checkpoints/`) and follow the README folder layout for training/eval datasets.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` sets up the runtime stack (PyTorch Lightning, Hydra, SMPL-X, Detectron2 dependencies).
- `python tokenization/train_poseVQ.py --cfg configs/tokenizer_amass_moyo.yaml` trains the pose tokenizer; update Hydra overrides before launching additional configs.
- `python tokenhmr/train.py datasets=mix_all experiment=tokenhmr_release` runs the full TokenHMR training pipeline (4DHumans + BEDLAM) using `tokenhmr/lib/configs_hydra/experiment/` defaults.
- `python tokenhmr/eval.py --dataset EMDB,3DPW-TEST --dataset_dir tokenhmr/dataset_dir/evaluation_data --checkpoint data/checkpoints/tokenhmr_model.ckpt --model_config data/checkpoints/model_config.yaml` evaluates checkpoints on 3DPW/EMDB benchmarks; adjust `--batch_size` and `--log_freq` as needed.
- `python tokenhmr/demo.py <image_or_video>` reuses the pretrained model to visualize mesh predictions on local media.

## Coding Style & Naming Conventions
- Follow the prevailing Python style in the repo: 4-space indentation, snake_case for functions, PascalCase for classes, and descriptive module-level docstrings aligned with Hydra/OmegaConf config names.
- Keep Hydra config names brief and lower case (e.g., `tokenhmr_release`, `tokenizer_amass_moyo`).
- Format code with `black`/`ruff` defaults if you add new files, and run `python -m compileall` only when importing new modules that require bytecode validation.

## Testing Guidelines
- No unit test suite is bundled; rely on script-based validation by running `demo.py`, `eval.py`, or a short training job after touching core logic.
- When you add new dataset splits or preprocessors, update/verify under `tokenhmr/dataset_dir/` and rerun the matching evaluation command to confirm output shapes.
- Name any test artifacts or checkpoints with the date and config (e.g., `tokenhmr_release_2024-10-AG1.ckpt`) so they are easy to trace in logs.

## Commit & Pull Request Guidelines
- Keep commit messages concise, include an optional scope tag in square brackets (e.g., `[Tokenization] Fix pose resampling`) and describe the change in an imperative tone, mirroring recent history.
- In PR descriptions list the datasets or configs affected, reference related issues/experiments, and highlight manual verification steps (training/eval/demo commands).
- Attach screenshots/videos for visual changes, mention large download steps, and confirm that `requirements.txt` stays in sync when adding new dependencies.

## Security & Configuration Tips
- Do not check in downloaded datasets or checkpoints; store them in `data/` outside version control and cite their source URLs when describing experiments.
- Sensitive access (project page registration) is required for BEDLAM/Tokenization data—note that in PRs when sharing reproduction steps.

## 说明与沟通
- 请使用中文进行所有文档和讨论性的解释，特别是在贡献说明、报错描述和会议记录中保持中文表达。
