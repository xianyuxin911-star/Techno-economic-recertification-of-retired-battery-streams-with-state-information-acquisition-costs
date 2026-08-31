# Part 4 — Fixed-scale techno-economic optimization

## Core role

Part 4 combines the Part 3 prediction-error model with chemistry-specific action values and reproduces the fixed-scale results in Main Figs. 2b–3f.

## Required inputs

```text
Input/LFP_SOH_Table.mat
Input/NMC_SOH_Table.mat
../Part3_Prediction error extrapolation/Output/Results/P3_Exp02_ResidualExtrapolation_Model.mat
Function/local_build_noise_by_k.m
```

## Run order and main-figure mapping

### 1. Fixed-scale LFP optimization

```matlab
P4_Exp01_fixed_scale_LFP_optimization
```

Produces the LFP result and workspace files used by later scripts.

Reproduces:

- Main Fig. 2b–e — fixed-scale LFP cost and decision analysis
- Main Fig. 3a–b — LFP net-value curves and recycling-value breakdown

### 2. Fixed-scale NMC optimization

```matlab
P4_Exp02_fixed_scale_NMC_optimization
```

Produces the NMC result and workspace files used by later scripts.

Reproduces:

- Main Fig. 3c–d — NMC net-value curves and recycling-value breakdown

### 3. Chemistry comparison

```matlab
P4_Exp03_chemistry_comparison_summary_plots
```

Reads the result files from Steps 1 and 2.

Reproduces:

- Main Fig. 3e–f — chemistry-driven action reallocation and value-gap comparison

## Downstream use

The fixed-scale workspace files are also required by `P5_Exp03_NMC_error_scaling_scale_robustness.m`, which produces Main Fig. 5d–f.

## Public-release boundary

Standalone fixed-k regret, true-SOH benchmark, label-cost sensitivity, refurbishment sensitivity, fixed-scale error-scaling, and gray-zone scripts are not included because they support Extended Data, Supplementary, or additional robustness analyses.

Generated `Output/` and `Figures/` directories are excluded from version control.
