# Python training entry point

`train_risk_estimator.py` is dependency-free and trains/calibrates the
incremental-harm estimator from `risk_training_dataset.csv`. It splits entire
scenario seeds, never individual packet rows.

```powershell
python train_risk_estimator.py ..\outputs\paired_externality_<run>\risk_training_dataset.csv `
  --output ..\models\risk_estimator.json
```

The held-out test seed is reported but must not be used to select features,
hyperparameters, or calibration thresholds.

The pre-training profile has one independent calibration seed. Its per-bin
maximum is the conservative engineering upper used by the artifact. The
Wilson value is retained only as an iid diagnostic because packet exposures
within a scenario are clustered; it is not a certified confidence bound.

`train_dqn.py` trains a six-action offline DQN (`reject`, `local`, and four
ranked helpers) from native `dqn_explore` trajectories. WiLab loads the JSON
network when `safetyOffloadingPolicy` is `dqn`; invalid helper ranks are masked.

```powershell
python train_dqn.py ..\outputs\paired_dqn_explore_<run>\risk_training_dataset.csv `
  --output ..\models\dqn_policy_v1.json
```
