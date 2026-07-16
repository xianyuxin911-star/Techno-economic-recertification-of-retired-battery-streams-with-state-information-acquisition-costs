# Part 2 — Target-domain Pack-level SOH Prediction

## Purpose

Part 2 transfers the source-domain cell-level CNN to target-domain battery packs. It compares zero-shot inference, frozen-encoder learning, encoder fine-tuning, L2-SP regularization, regression heads, attention-pooling strategies, and fine-tuning depths.

## Required inputs

```text
Target_Data/001PB*.mat
../Part1_Source-domain_Prediction/Output/Models/P1_SourceDomain_CNN_Model.mat
```

Run `P2_Exp01` first. Experiments `P2_Exp02`–`P2_Exp07` and `P2_Exp11` then use the processed target-pack dataset. Summary scripts must be run after the experiments that supply their inputs.

## Code-to-result map

### Step 1 — Construct the target-domain dataset

**Run**

```matlab
P2_Exp01_target_data_processing
```

**Reads:** `Target_Data/001PB*.mat`

**Produces:** `Output/Data/P2_Exp01_TargetPack_Input.mat`

**Reproduces**

> **Supplementary Fig. 1b — Target-domain Q–V curves**  
> `Figures/Supplementary/SupFig01b_target_domain_QV_curves.png`

The script extracts 45-channel pack voltage features, resamples each channel to 512 points, and saves the pack-level SOH labels and identifiers.

---

### Step 2 — Zero-shot Min/Worst-1 baseline

**Run:** `P2_Exp02_zero_shot_min_worst1_prediction`

**Reads:** the Part 1 CNN model and `P2_Exp01_TargetPack_Input.mat`

**Produces**

```text
Output/Results/P2_Exp02_ZeroShot_MinWorst1_{Metrics,PackPredictions,CellPredictions}.csv
Output/Results/P2_Exp02_ZeroShot_MinWorst1_Results.mat
```

**Figure use:** supplies the zero-shot results used in Extended Data Fig. 1b by `P2_Exp08`.

---

### Step 3 — Frozen-encoder transfer

**Run:** `P2_Exp03_frozen_encoder_attention_prediction`

**Produces**

```text
Output/Results/P2_Exp03_FrozenEncoder_{Metrics,PackPredictions}.csv
Output/Results/P2_Exp03_FrozenEncoder_Results.mat
```

**Figure use:** supplies the frozen-encoder results used in Extended Data Fig. 1b and the frozen-depth reference used by `P2_Exp11`.

---

### Step 4 — Encoder fine-tuning

**Run:** `P2_Exp04_fine_tuning_attention_prediction`

**Produces**

```text
Output/Results/P2_Exp04_FineTune_{Metrics,PackPredictions}.csv
Output/Results/P2_Exp04_FineTune_Results.mat
```

**Figure use:** supplies the fine-tuning results used in Extended Data Fig. 1b.

---

### Step 5 — L2-SP fine-tuning

**Run:** `P2_Exp05_fine_tuning_L2SP_attention_prediction`

**Produces**

```text
Output/Results/P2_Exp05_FineTuneL2SP_{Metrics,PackPredictions,CellAttention}.csv
Output/Results/P2_Exp05_FineTuneL2SP_Results.mat
```

**Figure use:** supplies the L2-SP results used in Extended Data Fig. 1b and the Last3Conv reference used by `P2_Exp11`.

---

### Step 6 — Compare regression heads

**Run:** `P2_Exp06_regression_head_comparison`

**Produces**

```text
Output/Results/P2_Exp06_RegressionHead_{Metrics,PackPredictions}.csv
Output/Results/P2_Exp06_RegressionHead_Results.mat
```

**Figure use:** supplies the MLP, GPR, and SVR predictions visualized as Supplementary Fig. 4a–c by `P2_Exp09`.

---

### Step 7 — Attention-pooling ablation

**Run:** `P2_Exp07_attention_pooling_ablation`

**Produces:** method-specific pack-prediction and cell-attention CSV files for Mean, Standard, Softmin, and Hardmin pooling, plus:

```text
Output/Results/P2_Exp07_AttentionPoolingAblation_Metrics.csv
Output/Results/P2_Exp07_AttentionPoolingAblation_Results.mat
```

**Figure use:** supplies the data visualized as Extended Data Fig. 1c–e by `P2_Exp10`.

---

### Step 8 — Transfer-strategy summary

**Run:** `P2_Exp08_transfer_strategy_comparison`

**Reads:** pack predictions from `P2_Exp02`–`P2_Exp05`

**Produces**

```text
Output/Results/P2_Exp08_TransferStrategy_Metrics.csv
Output/Results/P2_Exp08_TransferStrategy_PredictionsLong.csv
```

**Reproduces**

> **Extended Data Fig. 1b — Transfer-strategy prediction trajectories**  
> `Figures/Extended/ExD01b_TransferStrategyTracking.png`

The script also writes two non-manuscript diagnostic figures:

```text
Figures/Extended/P2_Exp08_TransferStrategyErrorBoxplot_diagnostic.png
Figures/Extended/P2_Exp08_TransferStrategyRMSESummary_diagnostic.png
```

---

### Step 9 — Regression-head summary

**Run:** `P2_Exp09_regression_head_summary_plots`

**Reads:** `P2_Exp06_RegressionHead_PackPredictions.csv`

**Produces:** regression-head summary metrics and prediction CSV files.

**Reproduces**

> **Supplementary Fig. 4a–c — Regression-head parity, R², and RMSE/MAPE comparisons**  
> `Figures/Supplementary/SupFig04a_regression_head_parity_plot.png`  
> `Figures/Supplementary/SupFig04b_regression_head_R2_gradient.png`  
> `Figures/Supplementary/SupFig04c_regression_head_RMSE_MAPE_lollipop.png`

---

### Step 10 — Attention-pooling summary

**Run:** `P2_Exp10_attention_pooling_summary_plots`

**Reads:** method-specific prediction and attention files from `P2_Exp07`

**Reproduces**

> **Extended Data Fig. 1c–e — Attention tracking, error heat map, and maximum-attention heat map**  
> `Figures/Extended/ExD01c_attention_tracking.png`  
> `Figures/Extended/ExD01d_attention_error_heatmap.png`  
> `Figures/Extended/ExD01e_max_attention_heatmap.png`

---

### Step 11 — Fine-tuning-depth ablation

**Run:** `P2_Exp11_FineTuneDepth_Ablation`

**Reads:** the processed target dataset, Part 1 model, and reference results from `P2_Exp03` and `P2_Exp05`

**Produces**

```text
Output/Results/P2_Exp11_FineTuneDepthAblation_Metrics.csv
Output/Results/P2_Exp11_FineTuneDepthAblation_SOHPredictions.csv
Output/Results/P2_Exp11_FineTuneDepthAblation_{Last1Conv,Last2Conv}_Attention.csv
Output/Results/P2_Exp11_FineTuneDepthAblation_Results.mat
```

This script exports numerical ablation results but does not write a manuscript figure.

## Repository policy

Raw target data and MATLAB scripts are version-controlled. Generated `Output/` and `Figures/` directories remain local and are excluded from GitHub.
