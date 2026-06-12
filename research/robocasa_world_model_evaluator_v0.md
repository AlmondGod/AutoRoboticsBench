# RoboCasa World-Model Evaluator v0

Goal:
- Replace most slow RoboCasa simulator evaluations with a fast learned evaluator.
- Use the simulator only to validate top candidates and measure evaluator correlation.

Long-term architecture:
- Teacher/init: Cosmos-Predict2.5 Video2World or another internet-video world model.
- Finetune target: RoboCasa-5 action-conditioned rollouts.
- Distilled evaluator: small latent dynamics model with progress, success, and latent prediction heads.

Implemented v0:
- `models/robocasa_tiny_evaluator.py`
  - Encodes two RGB camera views, proprio, and task id into a latent.
  - Predicts action-conditioned next latent and next proprio.
  - Predicts progress and success probability from latent state.
- `train/train_robocasa_tiny_evaluator.py`
  - Trains from RoboCasa demonstration videos/actions.
  - Treats late successful demo frames as positive success labels.
  - Saves normalization stats so candidate policies can be scored offline.
- `eval/eval_robocasa_tiny_evaluator_correlation.py`
  - Scores existing autoresearch policy candidates with imagined latent rollouts.
  - Writes learned-vs-sim candidate records.
  - Produces Pearson/Spearman/top-k ranking metrics and a correlation SVG.

Smoke result:
- Run: `runs/robocasa/world_evaluator/tiny_opendrawer_task0_stable_smoke`
- Data: OpenDrawer task-index 0, 80 train demos, held-out episodes 87/92/93/94/98/100/101.
- Training: 150 steps, frame stride 8, 957 train transitions, 158 validation transitions.
- Offline validation:
  - val loss: 0.1889
  - progress MAE: 0.0568
  - success-label accuracy: 0.968
- Candidate ranking over 38 archived task-0 policies:
  - learned rollout speed: ~741 imagined rollouts/sec
  - Pearson correlation with RoboCasa sim success: -0.307
  - Spearman correlation: -0.044
  - top-5 hit: 0.0

Interpretation:
- The evaluator is fast enough for the thesis.
- The current score is not useful yet because it only scores action chunks from the initial observation.
- Many archive rows differ only in commit horizon or temporal ensembling; those changes affect closed-loop execution but not the first predicted action chunk.
- Next v0.1 should record actual action traces during RoboCasa sim eval and train/score on those traces.

Current limitation:
- This v0 is a fast evaluator scaffold, not yet a full visual world model.
- It imagines in learned latent space using predicted policy action chunks from initial observations.
- It does not yet decode video frames or use Cosmos features.

Next pass:
- Add a Cosmos feature cache interface:
  - raw RGB frames -> Cosmos latent/features
  - train the tiny evaluator on Cosmos latents instead of its own small CNN latent
- Record policy rollout action traces during real sim eval.
  - This lets the evaluator score full closed-loop action sequences, not just initial action chunks.
- Validate correlation on a larger candidate set and across RoboCasa-5, not only OpenDrawer task-index 0.

v0.1 trace-evaluator result:
- Added action trace recording to `eval/eval_robocasa_policy_ensemble.py`.
  - Each sim eval can now write one `.npz` per episode with executed actions and success flags.
- Added trace scoring to `eval/eval_robocasa_tiny_evaluator_correlation.py`.
  - `--prefer-action-traces` scores the actual executed action sequence instead of only the first predicted chunk.
  - `--invert-learned-score` supports the observed calibration where the raw head acts like a failure/risk score on traces.
- Re-evaluated experiments 29-38 with action traces:
  - trace archive: `runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl`
  - calibrated correlation plot: `runs/robocasa/world_evaluator/trace_eval_frontier/correlation_trace_calibrated.svg`
- Trace-based calibrated evaluator:
  - n: 10 candidates
  - learned speed: ~243 imagined rollouts/sec at 260 imagined steps
  - Pearson correlation with sim success: 0.995
  - Spearman correlation: 0.931
  - top-5 hit: 1.0

Interpretation:
- Full action traces fix the main v0 failure mode.
- The tiny evaluator is now useful as a cheap pre-filter over candidate rollouts, as long as candidates provide planned/executed action traces.
- This is still not a no-sim replacement for policy evaluation because the trace currently comes from sim eval. The next step is to score proposed policy action traces before full sim validation, then run RoboCasa only on top-ranked candidates.

v0.2 VAE-latent visual world model:
- Replaced the bolt-on latent-to-RGB decoder with a VAE-style visual latent model in `models/robocasa_tiny_evaluator.py`.
  - Encoder: two RGB views, proprio, and task id -> Gaussian latent.
  - Decoder: latent and task id -> reconstructed left-view and wrist/right-view RGB.
  - Dynamics: latent, action, and task id -> next latent and next proprio.
  - Heads: progress and success/risk prediction.
- Added:
  - `train/train_robocasa_vae_world_model.py`
  - `train/train_robocasa_vae_trace_calibrator.py`
  - `eval/render_robocasa_vae_rgb_rollout.py`
- Base visual training:
  - run: `runs/robocasa/world_evaluator/vae_opendrawer_task0_smoke`
  - data: OpenDrawer task-index 0, 80 train demos, held-out episodes 87/92/93/94/98/100/101
  - train samples: 957
  - val samples: 158
  - train time: 23.5 sec on local MPS/CPU-auto setup
  - best val loss: 0.6006
  - val RGB PSNR: 14.62 dB
  - val RGB MSE: 0.0345
  - progress MAE: 0.0529
  - success-label accuracy: 0.962
- Trace calibration:
  - run: `runs/robocasa/world_evaluator/vae_opendrawer_task0_trace_calibrated_h260`
  - initialized from the base VAE checkpoint
  - frozen encoder and decoder, so reconstruction quality is preserved
  - trained only dynamics/progress/success scoring on archived policy action traces
  - rollout horizon: 260 steps
  - best trace loss: 0.0316
  - train time: 127.4 sec
- Candidate ranking with calibrated VAE score:
  - archive: `runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl`
  - plot: `runs/robocasa/world_evaluator/vae_opendrawer_task0_trace_calibrated_h260/correlation_trace_calibrated.svg`
  - n: 10 candidates
  - raw success head behaves like a failure/risk score, so evaluation uses `--invert-learned-score`
  - Pearson correlation with RoboCasa sim success: 0.9979
  - Spearman correlation: 0.9309
  - top-5 hit: 1.0
  - best learned candidate id: 32
  - best sim candidate id: 32
- Decoded rollout:
  - video: `runs/robocasa/world_evaluator/vae_opendrawer_task0_trace_calibrated_h260/vae_decoded_world_rollout_exp034_ep87.mp4`
  - source trace final sim success: true

Interpretation:
- VAE latents improve reconstruction over the previous bolt-on decoder, but the decoded video is still blurry and not yet high-fidelity enough to be a strong visual simulator.
- The same latent model can be trace-calibrated into a very strong candidate ranker on the current archive.
- The correlation result is not a held-out claim yet: the trace archive used for calibration is also the archive used for the reported ranking measurement.
- Next required step is a proper held-out candidate split: calibrate on one set of policy traces, then score unseen candidate traces before running RoboCasa sim validation.

v0.3 flow-matching next-RGB trial:
- Added a compact conditional rectified-flow decoder:
  - `models/robocasa_flow_rgb.py`
  - `train/train_robocasa_flow_next_rgb.py`
  - `eval/render_robocasa_flow_rgb_rollout.py`
- Goal:
  - improve the blurry VAE visual rollout by training a flow decoder for next-frame prediction.
  - keep the VAE encoder/dynamics/evaluator frozen so the existing correlation result is not disturbed.
- Trial A: noise -> next RGB.
  - run: `runs/robocasa/world_evaluator/flow_next_rgb_opendrawer_task0_smoke`
  - best step: 200
  - validation sampled PSNR: 11.31 dB
  - result: rejected; much worse than the VAE decoder baseline.
- Trial B: residual flow from VAE predicted RGB -> next RGB.
  - run: `runs/robocasa/world_evaluator/flow_next_rgb_opendrawer_task0_residual`
  - best step: 100
  - best validation sampled PSNR: 14.33 dB
  - final validation sampled PSNR: 14.16 dB
  - result: rejected; close to but still below the VAE decoder baseline of 14.62 dB.
- Visual comparison:
  - `runs/robocasa/world_evaluator/flow_next_rgb_opendrawer_task0_residual/real_vs_vae_vs_flow_exp034_ep87.mp4`

Interpretation:
- Flow matching did not improve reconstruction quality in this small setup.
- Residual flow is much better than noise flow, but it mostly preserves the VAE blur and sometimes adds noise/contrast rather than useful detail.
- A likely next attempt is not a tiny pixel-space flow head, but a stronger spatial decoder: skip-connected latent U-Net, patch-token diffusion/flow over VQ tokens, or training the encoder/decoder and dynamics jointly on multi-step video prediction.
