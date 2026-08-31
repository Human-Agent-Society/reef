
# Borden 3D Groundwater Contaminant Source Inversion

## Task
You are given a Borden-style 3D groundwater contaminant migration scene derived from a Borden-AdePy reproduction workflow. Your task is to infer a **finite-duration rectangular-region contaminant source** from public monitoring-well readings.

This is not a point-source task. A point-source answer or the old `x0,y0,z0,C0` schema is accepted only as a degenerate fallback and cannot receive full credit.

## Required source parameterization
Create `answer.json` in the task root using this schema:

```json
{
  "source_type": "rectangular_region",
  "dimension": 3,
  "x_center": 0.0,
  "y_center": 0.0,
  "z_center": 0.0,
  "half_length_x": 0.0,
  "half_length_y": 0.0,
  "half_length_z": 0.0,
  "C0": 0.0,
  "t_start": 0.0,
  "duration": 0.0,
  "transport_model": {
    "equation_type": "advection_dispersion_reaction",
    "governing_equation": "R*dC/dt = div(D grad C) - v dot grad C - lambda*C + source",
    "velocity_m_per_day": 0.0,
    "alpha_L_m": 0.0,
    "alpha_TH_m": 0.0,
    "alpha_TV_m": 0.0,
    "porosity": 0.0,
    "retardation_factor": 1.0,
    "lambda_per_day": 0.0,
    "numerical_approach": "brief description of forward model and optimization"
  },
  "method": "brief description of your inversion method"
}
```

- `x_center,y_center,z_center`: center of the rectangular source region, in meters.
- `half_length_x,half_length_y,half_length_z`: half sizes of the rectangular source region, in meters.
- `C0`: effective source concentration/intensity, in mg/L.
- `t_start`: release start time in days.
- `duration`: release duration in days.
- `transport_model`: your groundwater solute transport construction. Include the
  ADE/reaction governing equation, public hydrogeologic parameters used, and the
  numerical or analytical approximation used to predict concentrations.

All parameters must stay within `public_problem_config.json` → `source_search_bounds_for_agent`.

## Provided files

- `public_problem_config.json`: Borden grid, hydrogeological parameters, source prior bounds, and column definitions.
- `public_source_prior.json`: explicit range summary for the finite-duration rectangular-region source parameters.
- `borden_grid.npz`: grid arrays exported from the Borden-AdePy scene, including x/y grid and bottom profile.
- `public_wells.csv`: public monitoring-well coordinates.
- `public_monitoring_data.csv`: noisy, censored public monitoring observations. Clean generated
  concentrations are not provided to the agent.
- `baseline_solver.py`: writes a legal low-quality baseline `answer.json` from the center of the parameter bounds.
- `answer_template.json`: required output schema.

No runnable scientific starter solver is provided. You should write your own Python code to read files, construct the groundwater solute transport equation, build an ADE/AdePy or equivalent forward approximation, optimize the source parameters, and update `answer.json`.

## Mandatory baseline workflow

Before implementing a complex inversion algorithm, first ensure that a valid `answer.json` exists:

```bash
python baseline_solver.py
```

Then iteratively improve `answer.json` using public monitoring data. Missing `answer.json` gives zero.

## Scoring policy

The judge does not grade old point-source location error directly. It evaluates whether your finite-duration rectangular-region source predicts withheld monitoring readings:

1. Read `answer.json`.
2. Check required finite-region fields and parameter bounds.
3. Use a hidden Borden-ADE region-source forward model to predict concentrations at hidden monitoring wells and hidden future times.
4. Compare predictions with hidden readings using relative RMSE and log-scale metrics.
5. Hidden prediction dominates the score. Easy format/prior/method/transport-equation points are capped and cannot exceed the hidden-quality caps.
6. Region-shape and physical-consistency points are gated by hidden prediction quality.
7. Very poor hidden/future prediction is capped at 15 points; poor-but-improving prediction is capped at 30 points; moderate prediction is capped at 45 points.

If hidden monitoring prediction is poor, the final score is capped. This makes early baseline or shallow point-source fits low-scoring, while still allowing genuine improvement through better region-source inversion.

## Rules

- Do not read or reference hidden scoring files.
- Do not hard-code hidden monitoring readings.
- Do not modify judge files.
- Do not call FloPy, MODFLOW, MT3DMS, or external groundwater executables.
- You may use Python libraries such as NumPy, SciPy, pandas, matplotlib, pymoo, and AdePy if available.
- Submit all scripts/results needed to reproduce your `answer.json`.

## Critical submission rule

The judge only reads answer.json in the task root.

It does not automatically run inverse_solver.py, run_checks.py, write_fit_report.py,
write_public_predictions.py, or any other script during scoring.

Therefore, after every meaningful inversion or optimization step, you must immediately
overwrite the task-root answer.json with the best current source parameters.

If answer.json is unchanged, the score and all METRICS will remain unchanged, even if
you create new Python scripts, reports, or prediction files.

Recommended workflow:

1. Run python baseline_solver.py only as an initial fallback.
2. Implement and run your inversion script.
3. After each improved parameter set is found, write it to answer.json.
4. Run python validate_answer.py if available.
5. Submit only after confirming that answer.json has changed.
