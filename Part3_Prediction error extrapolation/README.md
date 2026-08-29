# Part 3 — Label-dependent prediction-error model

## Core role

Part 3 is the prediction-to-TEA bridge. It estimates how target-pack SOH prediction residuals change with the number of labelled packs and exports the uncertainty model used by Parts 4 and 5.

## Required inputs

```text
../Part2_Target-domain_Prediction/Output/Data/P2_Exp01_TargetPack_Input.mat
../Part1_Source-domain_Prediction/Output/Models/P1_SourceDomain_CNN_Model.mat
```

## Run order

```matlab
P3_Exp01_NestedResampling_ErrorEvaluation
P3_Exp03_label_dependent_residual_extrapolation
```

### `P3_Exp01_NestedResampling_ErrorEvaluation.m`

The script repeatedly trains the selected L2-SP predictor at different labelled-pack counts and evaluates residuals on fixed test packs.

Produces:

```text
Output/Results/P3_Exp01_NestedResampling_Checkpoint.mat
Output/Results/P3_Exp01_NestedResampling_Final.mat
Output/Results/P3_Exp01_NestedResampling_Summary.csv
```

This is the most computationally intensive step.

### `P3_Exp03_label_dependent_residual_extrapolation.m`

Reads:

```text
Output/Results/P3_Exp01_NestedResampling_Final.mat
```

Produces:

```text
Output/Results/P3_Exp03_ResidualExtrapolation_EmpiricalFeatures_k5_k21.csv
Output/Results/P3_Exp03_ResidualExtrapolation_Params_k5_k50.csv
Output/Results/P3_Exp03_CoreRegionFeatures_PM_PR_SigmaR_k5_k50.csv
Output/Results/P3_Exp03_ResidualExtrapolation_Model.mat
```

`P3_Exp03_ResidualExtrapolation_Model.mat` is the principal Part 3 output and is loaded directly by the fixed- and varied-scale TEA scripts.

## Public-release boundary

The standalone prediction-error summary and residual-region-boundary robustness scripts are not included because they produce secondary or supplementary results and are not inputs to `P3_Exp03`.

Generated `Output/` and `Figures/` directories are excluded from version control.
