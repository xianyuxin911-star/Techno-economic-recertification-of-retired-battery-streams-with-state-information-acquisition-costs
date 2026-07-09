# Part 5 — Economic Optimization under Varied Scale

## Purpose

Part 5 extends the recertification optimization across deployment scales. It evaluates scale-dependent label investment, cost amortization, chemistry-specific value, prediction-error robustness, fixed-\(k\) regret, and exploratory market-scale throughput.

## Required inputs

```text
Input/LFP_SOH_Table.mat
Input/NMC_SOH_Table.mat
../Part3_Prediction error extrapolation/Output/Results/P3_Exp03_ResidualExtrapolation_Model.mat
Function/local_build_noise_by_k.m
```

Run `P5_Exp01` before `P5_Exp02`. Scripts `P5_Exp03`–`P5_Exp06` consume results from these baseline analyses.

## Code-to-result map

### Step 1 — Scale-dependent LFP optimization

**Run:** `P5_Exp01_scale_dependent_LFP_optimization`

**Produces:** LFP result/workspace MAT files, scale summary, action-value decomposition, action-allocation data, and A3 inset data under `Output/Results/P5_Exp01_*`.

**Reproduces**

> **Main Fig. 4a–f — LFP scale-dependent label investment, costs, and action value**  
> `Figures/Main/MainFig04a_LFP_kstar_total_cost_landscape.png`  
> `Figures/Main/MainFig04b_LFP_label_density_per_ton.png`  
> `Figures/Main/MainFig04c_LFP_per_pack_cost_breakdown.png`  
> `Figures/Main/MainFig04d_LFP_per_pack_cost_reduction_decomposition.png`  
> `Figures/Main/MainFig04e_LFP_total_cost_breakdown.png`  
> `Figures/Main/MainFig04f_LFP_action_value_decomposition.png`
>
> **Extended Data Fig. 4c — LFP scale-dependent optimal thresholds**  
> `Figures/Extended/ExD04c_LFP_scale_dependent_optimal_thresholds.png`

The script also exports LFP action-allocation data used in Extended Data Fig. 5.

---

### Step 2 — Scale-dependent NMC optimization

**Run:** `P5_Exp02_scale_dependent_NMC_optimization`

**Reads:** the Part 3 residual model and the Step 1 LFP result file.

**Produces:** NMC result/workspace MAT files, scale summary, manuscript-value summary, action-value decomposition, and action-allocation data under `Output/Results/P5_Exp02_*`.

**Reproduces**

> **Main Fig. 5a–c — NMC cost reduction, total cost, and action value**  
> `Figures/Main/MainFig05a_NMC_per_pack_cost_reduction_decomposition.png`  
> `Figures/Main/MainFig05b_NMC_total_cost_breakdown.png`  
> `Figures/Main/MainFig05c_NMC_action_value_decomposition.png`
>
> **Extended Data Fig. 4d — NMC scale-dependent optimal thresholds**  
> `Figures/Extended/ExD04d_NMC_scale_dependent_optimal_thresholds.png`

The script also exports NMC action-allocation data used in Extended Data Fig. 5.

---

### Step 3 — NMC error-scaling robustness across deployment scales

**Run:** `P5_Exp03_NMC_error_scaling_scale_robustness`

**Reads:** Part 4 fixed-scale workspaces and `P5_Exp02_ScaleDependent_NMC_Results.mat`.

**Produces:** full and representative robustness CSV tables plus `P5_Exp03_NMC_ErrorScaling_ScaleRobustness_Results.mat`.

**Reproduces**

> **Main Fig. 5d–f — Optimal \(k\), label density, and per-pack cost under error scaling**  
> `Figures/Main/MainFig05d_NMC_kstar_under_error_scaling.png`  
> `Figures/Main/MainFig05e_NMC_label_density_under_error_scaling.png`  
> `Figures/Main/MainFig05f_NMC_per_pack_cost_under_error_scaling.png`

---

### Step 4 — LFP fixed-\(k\) benchmark

**Run:** `P5_Exp04_LFP_FixedKBenchmark_Heatmap`

**Reads:** `P5_Exp01_ScaleDependent_LFP_Results.mat`

**Produces**

```text
Output/Results/P5_Exp04_LFP_FixedKBenchmark_Table.csv
Output/Results/P5_Exp04_LFP_FixedKBenchmark_Data.mat
```

**Reproduces**

> **Main Fig. 4g — LFP fixed-\(k\) benchmark heat map**  
> `Figures/Main/MainFig04g_LFP_FixedKBenchmark_Heatmap.png`

---

### Step 5 — NMC fixed-\(k\) benchmark

**Run:** `P5_Exp05_NMC_FixedKBenchmark_Heatmap`

**Reads:** `P5_Exp02_ScaleDependent_NMC_Results.mat`

**Produces**

```text
Output/Results/P5_Exp05_NMC_FixedKBenchmark_Table.csv
Output/Results/P5_Exp05_NMC_FixedKBenchmark_Data.mat
```

**Reproduces**

> **Main Fig. 5g — NMC fixed-\(k\) benchmark heat map**  
> `Figures/Main/MainFig05g_NMC_FixedKBenchmark_Heatmap.png`

---

### Step 6 — Per-pack fixed-\(k\) regret

**Run:** `P5_Exp06_FixedKBenchmark_PerPack`

**Reads:** the Step 1 LFP and Step 2 NMC result files.

**Produces:** LFP/NMC per-pack benchmark tables and `P5_Exp06_FixedKBenchmark_PerPack_Data.mat`.

**Reproduces**

> **Extended Data Fig. 6a–b — LFP and NMC fixed-\(k\) regret per pack**  
> `Figures/Extended/ExD06a_LFP_FixedKBenchmark_PerPack.png`  
> `Figures/Extended/ExD06b_NMC_FixedKBenchmark_PerPack.png`

---

### Step 7 — Exploratory LFP market-scale extension

**Run:** `P5_Exp07_LFP_MarketScale_Extension`

**Reads:** the LFP SOH table and Part 3 residual model.

**Produces**

```text
Output/Results/P5_Exp07_LFP_MarketScale_Extension_Table.csv
Output/Results/P5_Exp07_LFP_MarketScale_CostBreakdown.csv
Output/Results/P5_Exp07_LFP_MarketScale_OnlyRows.csv
Output/Results/P5_Exp07_LFP_MarketScale_Extension_Results.mat
```

This script extends throughput scenarios to 500,000 tons. It is an exploratory numerical extension and does not write a manuscript figure.

## Figure-numbering note

`P5_Exp06` writes files labelled Extended Data Fig. 6a–b. Part 4 contains separate analyses with the same embedded figure numbers. The filenames above follow the scripts exactly and should be reconciled with the final manuscript numbering.

## Repository policy

Input SOH tables, functions, and MATLAB scripts are version-controlled. Generated `Output/` and `Figures/` directories remain local.
