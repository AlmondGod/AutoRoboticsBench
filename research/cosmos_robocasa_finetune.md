# Cosmos RoboCasa Finetune Plan

Goal:
- Finetune a Cosmos video world model on our RoboCasa data.
- Measure both visual prediction quality and correlation with real RoboCasa eval success.

Recommended model:
- First pass: `Cosmos-Predict2.5-2B-Video2World`.
- Avoid 14B until the 2B pipeline is working.
- For correlation/value prediction, prefer the Cosmos Policy recipe over plain Video2World because it predicts future images, proprio, actions, and value.

Relevant official docs:
- Cosmos Predict2.5 action-conditioned Video2World post-training:
  `https://docs.nvidia.com/cosmos/latest/predict2.5/post-training/video2world_action-conditioned.html`
- Cosmos Policy RoboCasa recipe:
  `https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2/cosmos_policy/post_training.html`

Local prep implemented:
- `data/export_robocasa_cosmos_action.py`
  - Exports our LeRobot RoboCasa demos into a Bridge-like Cosmos folder:
    - `videos/*.mp4`
    - `annotations/*.json`
    - `metas/*.txt`
    - `manifest.json`
  - Preserves native RoboCasa `action` as 12D.
  - Writes `state_full`, `state[:6]`, gripper state, reward/done, and prompt.
- `eval/eval_cosmos_video_quality.py`
  - Scores generated videos for visual quality.
  - If reference videos exist, computes PSNR.
  - Correlates generated-video scores with real RoboCasa sim success archive.

Export command:

```bash
PYTHONPATH=.:third_party/robocasa:third_party/robosuite \
/opt/anaconda3/envs/robocasa/bin/python data/export_robocasa_cosmos_action.py \
  --manifest data/robocasa5/manifest.json \
  --task-alias OpenDrawer \
  --robocasa-task-index 0 \
  --max-demos-per-task 80 \
  --out-dir data/cosmos_robocasa_action/opendrawer_task0 \
  --layout side_by_side \
  --resize 320 \
  --fps 20
```

Cosmos setup on GPU host:

```bash
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
cd cosmos-predict2.5
huggingface-cli login
python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B
```

Official action-conditioned post-training command shape:

```bash
torchrun --nproc_per_node=1 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_reason_embeddings_rectified_flow_2b_256_320 \
  ~dataloader_train.dataloaders
```

Needed Cosmos-side edit:
- The official action-conditioned example uses Bridge-style action annotations.
- Bridge action is 7D.
- RoboCasa action here is 12D.
- Cosmos config / conditioner must either:
  - set `action_dim=12`, or
  - map RoboCasa actions to a 7D end-effector action representation.

Evaluation after Cosmos generations:

```bash
PYTHONPATH=. /opt/anaconda3/envs/robocasa/bin/python eval/eval_cosmos_video_quality.py \
  --generations runs/robocasa/cosmos_generations/opendrawer_task0 \
  --sim-archive runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl \
  --out runs/robocasa/cosmos_eval/opendrawer_task0/candidate_scores.jsonl \
  --metrics-out runs/robocasa/cosmos_eval/opendrawer_task0/correlation_metrics.json \
  --plot runs/robocasa/cosmos_eval/opendrawer_task0/correlation.svg
```

Current blocker:
- GPU host setup succeeded:
  - Host: `root@216.81.245.138 -p 12909`
  - GPU: A100-SXM4-80GB
  - Workspace: `/workspace/robot-autoresearch-cosmos`
  - PyTorch upgraded from 2.4.1 to 2.6.0+cu124 because Diffusers Cosmos 2.5 import fails on 2.4.1.
  - `diffusers==0.38.0` imports `Cosmos2_5_PredictBasePipeline` successfully.
- RoboCasa Cosmos smoke data was exported on the host:
  - `data/cosmos_robocasa_action/opendrawer_task0_cosmos_smoke`
  - 10 OpenDrawer task-index-0 clips
  - side-by-side left/right camera, 224 px per view, 16 fps
  - `videos/*.mp4`, `metas/*.txt`, `annotations/*.json`, `metadata.csv`
- Smallest practical Cosmos 2.5 finetune command attempted:

```bash
HF_HOME=/workspace/hf_home CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision=bf16 \
  third_party/diffusers_cosmos/train_cosmos_predict25_lora.py \
  --pretrained_model_name_or_path nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --train_data_dir data/cosmos_robocasa_action/opendrawer_task0_cosmos_smoke \
  --output_dir runs/robocasa/cosmos25_lora/opendrawer_task0_rank4_smoke \
  --train_batch_size 1 \
  --num_train_epochs 1 \
  --checkpointing_epochs 1 \
  --seed 0 \
  --height 224 --width 448 --num_frames 49 \
  --allow_tf32 --gradient_checkpointing \
  --lora_rank 4 --lora_alpha 4 \
  --dataloader_num_workers 0 \
  --report_to tensorboard \
  --num_inference_steps 8 \
  --do_final_eval
```

- Current blocker is Hugging Face authorization, not code/GPU:
  - `nvidia/Cosmos-Predict2.5-2B` returns `403 Forbidden`.
  - Message: token is authenticated but "not in the authorized list".
  - Visit `https://huggingface.co/nvidia/Cosmos-Predict2.5-2B` and request/accept access with the same HF account, then rerun the command above.
- Also checked older Cosmos base checkpoints:
  - `nvidia/Cosmos-Predict2-2B-Video2World`: README visible, weights/config gated.
  - `nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned`: README visible, weights/config gated.
- `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B` metadata and a 3.9GB `.pt` checkpoint appear downloadable, but that is the Cosmos Policy path, not the requested Cosmos 2.5 Video2World LoRA finetune path.

Update after access was granted:
- `nvidia/Cosmos-Predict2.5-2B` downloaded and loaded successfully on the A100 host.
- The upstream Diffusers example required small compatibility patches for `diffusers==0.38.0`:
  - wrap the transformer with PEFT `get_peft_model(...)` instead of `dit.add_adapter(...)`
  - use the standard linear warmup scheduler signature
  - stringify non-scalar TensorBoard hparams
  - provide a local one-channel condition mask helper
  - save the final adapter with PEFT `save_pretrained(...)`
- Smoke finetune completed:
  - data: 10 OpenDrawer task-index-0 clips
  - model: Cosmos-Predict2.5-2B frozen base + rank-4 LoRA
  - trainable params: about 0.01B
  - resolution: 224x448, 49 frames
  - training: 10 optimizer steps, about 19 seconds after model load
  - losses: noisy because this is only 10 clips/steps, roughly 4.49 to 24.0 step loss
  - adapter: `/workspace/robot-autoresearch-cosmos/runs/robocasa/cosmos25_lora/opendrawer_task0_rank4_smoke/adapter_model.safetensors`
- Inference succeeded:
  - conditioned on first 5 real demo frames
  - generated 49 frames with 8 denoising steps
  - local generated video: `runs/robocasa/cosmos25_lora/opendrawer_task0_rank4_smoke_eval/cosmos_lora_ep000000.mp4`
  - local side-by-side: `runs/robocasa/cosmos25_lora/opendrawer_task0_rank4_smoke_eval/reference_vs_cosmos_lora_ep000000.mp4`

Expected resource shape:
- Cosmos 2B is roughly 2B parameters.
- Inference/post-training needs a large CUDA GPU; NVIDIA docs list tens of GB of VRAM for 2B Video2World.
- Use H100/H200/A100 80GB class hardware for the clean path.

First acceptance criteria:
- Visual: generated videos are nonblank, temporally coherent, and qualitatively resemble RoboCasa camera views.
- Offline visual metrics: PSNR if paired references are available, plus motion/sharpness/nonblank scores.
- Correlation: generated-video/value score should rank the known trace candidates similarly to real RoboCasa success.
- Beat the tiny evaluator baseline only if correlation is positive on held-out candidate policies.
