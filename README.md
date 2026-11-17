# Safety-preserving NR-V2X Mode-2 offloading pipeline

This repository is the minimal working project extracted from
`MinjaeChoi119/C-V2X-DRL-Resource-Management`. It contains the WiLabV2XSim
runtime needed by the native NR-V2X path, the safety/offloading overlay,
experiment runners, tests, configurations, and risk-estimator training code.
Unrelated congestion-control experiments, saved agents, paper scripts,
IEEE 802.11p/coexistence modules, posters, and generated outputs are excluded.

## Layout

- `MainFiles*`, `MatFiles*`: required WiLabV2XSim NR-V2X runtime.
- `ConfigFiles/mode2_pipeline.cfg`: Mode-2 simulator configuration.
- `SafetyOffloading/`: project implementation, runners, tests, and Python code.
- `docs/`: system specification and parameter provenance.
- `LICENSE-WiLabV2XSim`: upstream simulator license.

## Verification

```matlab
cd SafetyOffloading
results = runtests("tests");
assertSuccess(results);
summary = runWiLabIntegratedSmoke();
```

## Pre-training data

```matlab
validation = runWiLabSafetyValidation("config/training_data.json");
[externality,riskRows] = runWiLabPairedExternality( ...
    "config/training_data.json",0.2,[1 2 3]);
```

Train the risk estimator with Python 3.10 or newer:

```powershell
python SafetyOffloading/python/train_risk_estimator.py `
  SafetyOffloading/outputs/<paired-run>/risk_training_dataset.csv `
  --output SafetyOffloading/models/risk_estimator.json
```

See `SafetyOffloading/README.md` for details and limitations.
