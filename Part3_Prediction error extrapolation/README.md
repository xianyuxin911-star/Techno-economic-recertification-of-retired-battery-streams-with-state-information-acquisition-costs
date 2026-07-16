# Part 3 — Prediction-error Extrapolation

## Purpose

Part 3 quantifies how pack-level SOH prediction error changes with the number of labelled target packs, fits a label-dependent residual model, and exports the uncertainty model used by the economic optimization in Parts 4 and 5.

## Required inputs

```text
../Part2_Target-domain_Prediction/Output/Data/P2_Exp01_TargetPack_Input.mat
../Part1_Source-domain_Prediction/Output/Models/P1_SourceDomain_CNN_Model.mat
```

Run the four scripts in numerical order.

## Code-to-result map

### Step 1 — Nested-resampling error evaluation

**Run:** `P3_Exp01_NestedResampling_ErrorEvaluation`

**Produces**

```text
Output/Results/P3_Exp01_NestedResampling_Checkpoint.mat
Output/Results/P3_Exp01_NestedResampling_Final.mat
Output/Results/P3_Exp01_NestedResampling_Summary.csv
```

The script repeatedly samples training sets at different labelled-pack counts \(k\), trains the L2-SP model, and evaluates residuals on fixed test packs. It is the computationally intensive base experiment for all later Part 3 analyses.

---

### Step 2 — Summarize label-dependent prediction errors

**Run:** `P3_Exp02_label_dependent_error_summary`

**Reads:** `P3_Exp01_NestedResampling_Final.mat`

**Produces:** `Output/Results/P3_Exp02_LabelDependentErrorSummary_Metrics.csv`

**Reproduces**

> **Extended Data Fig. 2b–d — Error decay, RMSE distributions, and prediction clouds**  
> `Figures/Extended/ExD02b_label_dependent_RMSE_MAE_decay.png`  
> `Figures/Extended/ExD02c_representative_RMSE_distribution.png`  
> `Figures/Extended/ExD02d_source_prediction_cloud_few_labels.png`  
> `Figures/Extended/ExD02d_source_prediction_cloud_medium_labels.png`  
> `Figures/Extended/ExD02d_source_prediction_cloud_large_labels.png`

---

### Step 3 — Fit and extrapolate the residual model

**Run:** `P3_Exp03_label_dependent_residual_extrapolation`

**Reads:** `P3_Exp01_NestedResampling_Final.mat`

**Produces**

```text
Output/Results/P3_Exp03_ResidualExtrapolation_EmpiricalFeatures_k5_k21.csv
Output/Results/P3_Exp03_ResidualExtrapolation_Params_k5_k50.csv
Output/Results/P3_Exp03_CoreRegionFeatures_PM_PR_SigmaR_k5_k50.csv
Output/Results/P3_Exp03_ResidualExtrapolation_Model.mat
```

**Reproduces**

> **Extended Data Fig. 3b–e — Fitted residual-feature scaling relationships**  
> `Figures/Extended/ExD03b_PM_fit.png`  
> `Figures/Extended/ExD03c_sigmaR_fit.png`  
> `Figures/Extended/ExD03d_PR_fit.png`  
> `Figures/Extended/ExD03e_RMSE_scaling_fit.png`
>
> **Supplementary Fig. 7 — Empirical and bridged residual distributions across k**  
> `Figures/Supplementary/SupFig07_residual_distributions_k05_to_k25/`

Selected distributions from Supplementary Fig. 7 are the source plots used to assemble Extended Data Fig. 3a.

`P3_Exp03_ResidualExtrapolation_Model.mat` is the principal Part 3 output consumed by Parts 4 and 5.

---

### Step 4 — Residual-region boundary robustness

**Run:** `P3_Exp04_ResidualRegionBoundaryRobustness`

**Reads:** `P3_Exp01_NestedResampling_Final.mat`

**Produces:** fit summaries, feature curves, fit curves, trend checks, sample counts, and the robustness model under `Output/Results/P3_Exp04_*`.

**Reproduces**

> **Supplementary Figs. 8–11 — Residual-region boundary robustness summaries**  
> `Figures/Supplementary/SupFig08_ResidualRegionBoundary_PM_sensitivity.png`  
> `Figures/Supplementary/SupFig09_ResidualRegionBoundary_PR_sensitivity.png`  
> `Figures/Supplementary/SupFig10_ResidualRegionBoundary_SigmaR_sensitivity.png`  
> `Figures/Supplementary/SupFig11_ResidualRegionBoundary_R2_summary.png`

Additional scenario-specific residual distributions are written to:

```text
Figures/Supplementary/SupFig12_residual_region_distributions_empirical_k05_to_k20/
```

## Repository policy

Part 3 generated files are not stored on GitHub. They are recreated locally under `Output/` and `Figures/`.
