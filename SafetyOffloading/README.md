# Safety-preserving NR-V2X Mode-2 offloading

This directory implements the system model in
`TITS_Mode2_Safety_Preserving_Offloading_Implementation_Spec.md` as a
standalone MATLAB experiment layer within the WiLabV2XSim tree.

The engine models:

- periodic safety traffic and optional computing traffic in one sidelink pool;
- sensing-based Mode-2 resource selection and semi-persistent reservations;
- packet fragmentation, co-channel interference, SINR/PER reception, and
  half-duplex loss;
- heterogeneous non-preemptive EDF CPU queues;
- local execution and the remote input -> compute -> result lifecycle;
- no-offload, unrestricted, EDF, CBR-threshold, priority-only, and risk-aware
  policies;
- matched baseline/treatment worlds and harmful-flip accounting;
- the structured CSV logs specified by the research design.

## Run

From MATLAB:

```matlab
cd WiLabV2Xsim-main_custom/SafetyOffloading
summary = runSafetyOffloadingExperiment("config/literature_grounded.json");
```

For a quick end-to-end run:

```matlab
summary = runSafetyOffloadingExperiment("config/smoke.json");
```

To run the application overlay through WiLabV2XSim's NR event loop, Mode-2
scheduler, SINR calculation, and PHY receiver-fate reporting:

```matlab
summary = runWiLabIntegratedSmoke();
```

Run the complete density, seed, and policy matrix from a JSON profile with:

```matlab
summary = runWiLabIntegratedExperiment("config/literature_grounded.json");
```

The integrated path tags regular WiLab safety packets, injects computing
fragments only through free application buffers, leaves resource reservations
under WiLab control, and consumes receiver outcomes from `updateKPICV2X`.

Outputs are written under `outputs/<run-id>/`. Run the acceptance suite with:

```matlab
results = runtests("tests");
assertSuccess(results);
```

Generate pre-training native data with:

```matlab
validation = runWiLabSafetyValidation("config/training_data.json");
[externality, riskRows] = runWiLabPairedExternality("config/training_data.json",0.2,[1 2 3]);
```

Validate the seed-separated Python training input without fitting a model:

```powershell
..\..\..\tools\python312\python.exe python\train_risk_estimator.py `
  outputs\paired_externality_20261117_225105\risk_training_dataset.csv --validate-only
```

All random reception decisions use stable packet/receiver keys. The paired
worlds therefore share exogenous mobility, task releases, safety releases,
fading samples, and decoding uniforms while allowing radio reservations and
collisions to diverge as endogenous effects.

