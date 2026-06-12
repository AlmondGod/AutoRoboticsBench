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
