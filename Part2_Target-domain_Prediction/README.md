# Part 2 — Target-domain data and selected predictor

## Core role

Part 2 retains only the target-pack data preparation and the selected L2-SP transfer-learning predictor.

## Run order

```matlab
P2_Exp01_target_data_processing
P2_Exp02_fine_tuning_L2SP_attention_prediction
```

### `P2_Exp01_target_data_processing.m`

Reads:

```text
Target_Data/001PB*.mat
```

Produces:

```text
Output/Data/P2_Exp01_TargetPack_Input.mat
```

This processed target-pack dataset is the required input for the nested-resampling experiment in Part 3.

### `P2_Exp02_fine_tuning_L2SP_attention_prediction.m`

Reads:

```text
Output/Data/P2_Exp01_TargetPack_Input.mat
../Part1_Source-domain_Prediction/Output/Models/P1_SourceDomain_CNN_Model.mat
```

Produces the selected L2-SP target-domain prediction results under `Output/Results/`.

`P2_Exp02` documents the final transfer-learning method. It is not an intermediate-file dependency of Part 3: `P3_Exp01` applies the same L2-SP formulation directly inside nested resampling so that label-dependent residuals can be estimated.

## Public-release boundary

Alternative zero-shot, frozen-encoder, unregularized fine-tuning, regression-head, attention-pooling, transfer-strategy, and fine-tuning-depth scripts are not included because they support comparison, ablation, Extended Data, or Supplementary results rather than the main-figure TEA chain.

Generated `Output/` and `Figures/` directories are excluded from version control.
