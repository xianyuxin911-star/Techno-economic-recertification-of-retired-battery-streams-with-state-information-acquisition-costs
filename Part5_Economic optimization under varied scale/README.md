# Part 5 — Varied-scale techno-economic optimization

## Core role

Part 5 reproduces the scale-dependent LFP and NMC results in Main Figs. 4a–5g.

## Required inputs

```text
Input/LFP_SOH_Table.mat
Input/NMC_SOH_Table.mat
../Part3_Prediction error extrapolation/Output/Results/P3_Exp03_ResidualExtrapolation_Model.mat
Function/local_build_noise_by_k.m
```

`P5_Exp03` additionally reads the fixed-scale workspaces produced by `P4_Exp01` and `P4_Exp02`.

## Run order and main-figure mapping

### 1. Scale-dependent LFP optimization

```matlab
P5_Exp01_scale_dependent_LFP_optimization
```

Reproduces Main Fig. 4a–f.

### 2. Scale-dependent NMC optimization

```matlab
P5_Exp02_scale_dependent_NMC_optimization
```

Reads the LFP scale list and results from Step 1 and reproduces Main Fig. 5a–c.

### 3. NMC error-scaling analysis

```matlab
P5_Exp03_NMC_error_scaling_scale_robustness
```

Reads the fixed-scale Part 4 workspaces and the NMC varied-scale result from Step 2. Reproduces Main Fig. 5d–f.

### 4. LFP fixed-k benchmark

```matlab
P5_Exp04_LFP_FixedKBenchmark_Heatmap
```

Reads the result from Step 1 and reproduces Main Fig. 4g.

### 5. NMC fixed-k benchmark

```matlab
P5_Exp05_NMC_FixedKBenchmark_Heatmap
```

Reads the result from Step 2 and reproduces Main Fig. 5g.

## Public-release boundary

The per-pack Extended Data benchmark and exploratory 300,000–500,000-ton market-scale extension are not included because they do not generate main-figure results.

Generated `Output/` and `Figures/` directories are excluded from version control.
