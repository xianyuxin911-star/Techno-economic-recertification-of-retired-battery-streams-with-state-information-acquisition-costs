# Part 4 — Economic Optimization under Fixed Scale

## Purpose

Part 4 combines the Part 3 prediction-error model with LFP- and NMC-specific action values. It optimizes the labelled-pack count \(k\) and predicted-SOH thresholds \(T_1\) and \(T_2\), then evaluates chemistry effects, regret, testing cost, refurbishment improvement, error scaling, and gray-zone tolerance.

## Required inputs

```text
Input/LFP_SOH_Table.mat
Input/NMC_SOH_Table.mat
../Part3_Prediction error extrapolation/Output/Results/P3_Exp03_ResidualExtrapolation_Model.mat
Function/local_build_noise_by_k.m
```

Run `P4_Exp01` and `P4_Exp02` first. Most later scripts consume their result or workspace files.

## Code-to-result map

### Step 1 — Fixed-scale LFP optimization

**Run:** `P4_Exp01_fixed_scale_LFP_optimization`

**Produces**

```text
Output/Results/P4_Exp01_FixedScale_LFP_{Workspace,Results}.mat
Output/Results/P4_Exp01_FixedScale_LFP_Summary.csv
```

**Reproduces**

> **Main Fig. 2b–e — LFP cost decomposition, threshold landscapes, confusion matrices, and value-loss matrices**  
> `Figures/Main/MainFig02b_fixed_scale_LFP_cost_decomposition.png`  
> `Figures/Main/MainFig02c_threshold_cost_landscapes/`  
> `Figures/Main/MainFig02d_confusion_matrices/`  
> `Figures/Main/MainFig02e_value_loss_matrices/`
>
> **Main Fig. 3a–b — LFP net-value curves and recycling-value breakdown**  
> `Figures/Main/MainFig03a_LFP_net_value_curves.png`  
> `Figures/Main/MainFig03b_LFP_recycling_breakdown.png`

---

### Step 2 — Fixed-scale NMC optimization

**Run:** `P4_Exp02_fixed_scale_NMC_optimization`

**Produces**

```text
Output/Results/P4_Exp02_FixedScale_NMC_{Workspace,Results}.mat
Output/Results/P4_Exp02_FixedScale_NMC_Summary.csv
```

**Reproduces**

> **Main Fig. 3c–d — NMC net-value curves and recycling-value breakdown**  
> `Figures/Main/MainFig03c_NMC_net_value_curves.png`  
> `Figures/Main/MainFig03d_NMC_recycling_breakdown.png`
>
> **Supplementary Fig. 15 — NMC fixed-scale cost decomposition**  
> `Figures/Supplementary/SupFig15_NMC_fixed_scale_cost_decomposition.png`

---

### Step 3 — Compare LFP and NMC pathways

**Run:** `P4_Exp03_chemistry_comparison_summary_plots`

**Reads:** the LFP and NMC result MAT files from Steps 1 and 2.

**Produces:** boundary-crossing, representative-SOH, transition-matrix, and action-share CSV tables under `Output/Results/P4_Exp03_*`.

**Reproduces**

> **Main Fig. 3e–f — Chemistry-driven action reallocation and refurbishment–recycling value gap**  
> `Figures/Main/MainFig03e_LFP_NMC_action_share_reallocation.png`  
> `Figures/Main/MainFig03f_refurbishment_recycling_value_gap.png`

---

### Step 4 — Fixed-scale LFP label-count regret

**Run:** `P4_Exp04_fixed_scale_LFP_k_regret_benchmark`

**Reads:** the Step 1 LFP results/workspace.

**Produces:** `P4_Exp04_FixedScale_LFP_KRegretBenchmark_*` result, curve, and scenario files.

**Reproduces**

> **Extended Data Fig. 6a — LFP fixed-scale \(k\)-regret benchmark**  
> `Figures/Extended/ExD06a_LFP_FixedScale_KRegretBenchmark.png`

---

### Step 5 — Fixed-scale NMC label-count regret

**Run:** `P4_Exp05_fixed_scale_NMC_k_regret_benchmark`

**Reads:** the Step 2 NMC results/workspace.

**Produces:** `P4_Exp05_FixedScale_NMC_KRegretBenchmark_*` result, curve, and scenario files.

**Reproduces**

> **Extended Data Fig. 6b — NMC fixed-scale \(k\)-regret benchmark**  
> `Figures/Extended/ExD06b_NMC_FixedScale_KRegretBenchmark.png`

---

### Step 6 — True-SOH action-value benchmark

**Run:** `P4_Exp06_TrueSOH_ActionValueBenchmark`

**Reads:** the LFP and NMC SOH input tables.

**Produces:** action-value curves, thresholds, and benchmark MAT results under `Output/Results/P4_Exp06_*`.

**Reproduces**

> **Extended Data Fig. 4a–b — True-SOH action-value benchmarks**  
> `Figures/Extended/ExD04a_LFP_true_SOH_action_value_benchmark.png`  
> `Figures/Extended/ExD04b_NMC_true_SOH_action_value_benchmark.png`

---

### Step 7 — LFP SOH-label-cost sensitivity

**Run:** `P4_Exp07_LFP_SOH_Label_Cost_Sensitivity.m`

**Reads:** `P4_Exp01_FixedScale_LFP_Results.mat`, with the corresponding workspace file accepted as a fallback.

**Produces:** summary, all-k, and MAT result files under `Output/Results/P4_Exp07_*`.

**Reproduces:**

- **Extended Data Fig. 7a:** LFP SOH-label-cost sensitivity  
  `Figures/Extended/ExD07a_LFP_SOH_label_cost_sensitivity.png`

- **Extended Data Fig. 7b:** Optimized cost decomposition under varying SOH-label costs  
  `Figures/Extended/ExD07b_LFP_SOH_label_cost_decomposition.png`

---

### Step 8 — LFP refurbishment-improvement sensitivity

**Run:** `P4_Exp08_LFP_Refurbishment_Improvement_Sensitivity`

**Reads:** `P4_Exp01_FixedScale_LFP_Workspace.mat`

**Produces:** scenario summaries, all-\(k\) results, prediction cache, and result MAT files under `Output/Results/P4_Exp08_*`.

**Reproduces**

> **Extended Data Fig. 9a–c — Action shares, thresholds, and cost decomposition**  
> `Figures/Extended/ExD09a_LFP_refurbishment_improvement_action_shares.png`  
> `Figures/Extended/ExD09b_LFP_refurbishment_improvement_thresholds.png`  
> `Figures/Extended/ExD09c_LFP_refurbishment_improvement_cost_decomposition.png`

---

### Step 9 — Fixed-scale NMC error-scaling robustness

**Run:** `P4_Exp09_FixedScale_NMC_ErrorScalingRobustness`

**Reads:** `P4_Exp02_FixedScale_NMC_Workspace.mat`

**Produces:** robustness summary, result MAT, and threshold-surface MAT files under `Output/Results/P4_Exp09_*`.

**Reproduces**

> **Extended Data Fig. 8a–e — Prediction error, decision loss, value retention, policy turnover, and action allocation**  
> `Figures/Extended/ExD08a_NMC_error_scaling_prediction_error_metrics.png`  
> `Figures/Extended/ExD08b_NMC_error_scaling_decision_loss_decomposition.png`  
> `Figures/Extended/ExD08c_NMC_error_scaling_net_value_retention.png`  
> `Figures/Extended/ExD08d_NMC_error_scaling_policy_turnover.png`  
> `Figures/Extended/ExD08e_NMC_error_scaling_action_allocation.png`

The optimized-threshold surface is retained as a non-manuscript diagnostic:

```text
Figures/Extended/P4_Exp09_NMC_error_scaling_optimized_thresholds_diagnostic.png
```

---

### Step 10 — LFP gray-zone tolerance

**Run:** `P4_Exp10_LFP_GrayZoneToleranceMechanism`

**Reads:** `P4_Exp01_FixedScale_LFP_Workspace.mat`

**Produces:** fixed-policy, re-optimized, and comparison tables for \(\eta_1\) and \(\eta_2\), plus a result MAT file.

**Reproduces**

> **Supplementary Fig. 16a–c — Eta1 gray-zone trigger rates, penalty costs, and policy re-optimization**  
> `Figures/Supplementary/SupFig16a_LFP_gray_zone_Eta1_trigger_rate_composition.png` through  
> `Figures/Supplementary/SupFig16c_LFP_gray_zone_fixed_vs_reoptimized_Eta1.png`
>
> **Supplementary Fig. 17a–c — Eta2 gray-zone trigger rates, penalty costs, and policy re-optimization**  
> `Figures/Supplementary/SupFig17a_LFP_gray_zone_Eta2_trigger_rate_composition.png` through  
> `Figures/Supplementary/SupFig17c_LFP_gray_zone_fixed_vs_reoptimized_Eta2.png`


## Extended Data Fig. 6 panel assignment

The panels of Extended Data Fig. 6 are assigned as follows:

- **Extended Data Fig. 6a:** LFP fixed-scale fixed-k benchmark (`P4_Exp04`)
- **Extended Data Fig. 6b:** NMC fixed-scale fixed-k benchmark (`P4_Exp05`)
- **Extended Data Fig. 6c:** LFP scale-dependent fixed-k additional cost per pack (`P5_Exp06`)
- **Extended Data Fig. 6d:** NMC scale-dependent fixed-k additional cost per pack (`P5_Exp06`)

The figure identifiers used by the current MATLAB scripts are consistent with this panel assignment.

## Repository policy

Input SOH tables, functions, and MATLAB scripts are version-controlled. Generated `Output/` and `Figures/` directories remain local.
