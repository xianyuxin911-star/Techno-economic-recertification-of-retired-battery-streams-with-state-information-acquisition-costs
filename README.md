# Techno-economic recertification of chemical and asset streams from retired batteries with state-information acquisition costs

This repository contains the minimal MATLAB workflow needed to connect cell-to-pack state-of-health (SOH) prediction with the techno-economic analysis (TEA) reported in the main figures.

## Public-release scope

The public repository intentionally retains only:

- the upstream prediction and prediction-error calculations required by the TEA;
- the fixed- and varied-scale optimization scripts that produce Main Figs. 2b–5g;
- the input data and shared functions required by those scripts.

Conceptual illustrations, including Main Fig. 2a, are not code-generated. Standalone scripts used only for Extended Data figures, Supplementary figures, ablations, alternative prediction baselines, additional sensitivity analyses, or exploratory market-scale extensions are outside this public-release scope.

## Computational chain

```text
Source-domain battery data
        |
        v
P1_Exp01: source-data preprocessing
        |
        v
P1_Exp02: source CNN training
        |
        +-------------------------------+
        |                               |
        v                               v
P2_Exp01: target-data preparation   P2_Exp02: selected L2-SP predictor
        |
        v
P3_Exp01: nested-resampling L2-SP prediction and error evaluation
        |
        v
P3_Exp02: label-dependent residual extrapolation
        |
        v
P3_Exp02_ResidualExtrapolation_Model.mat
        |
        +-------------------------------+
        |                               |
        v                               v
Part 4: fixed-scale TEA           Part 5: varied-scale TEA
        |                               |
        +---------------+---------------+
                        v
                  Main Figs. 2b–5g
```

`P2_Exp02` is retained as the transparent implementation of the selected L2-SP transfer-learning predictor. The error model used by the TEA is generated independently by `P3_Exp01`, which applies the same L2-SP approach inside nested resampling.

## Retained scripts and main-figure mapping

| Part | Retained script | Role or main-figure output |
|---|---|---|
| 1 | `P1_Exp01_SourceDomain_QVPreprocessing.m` | Prepare the source-domain CNN dataset |
| 1 | `P1_Exp02_SourceDomain_CNNTraining_CV.m` | Train the source CNN used by Part 3 |
| 2 | `P2_Exp01_target_data_processing.m` | Prepare the target-pack dataset used by Part 3 |
| 2 | `P2_Exp02_fine_tuning_L2SP_attention_prediction.m` | Selected target-domain predictor |
| 3 | `P3_Exp01_NestedResampling_ErrorEvaluation.m` | Generate label-dependent prediction residuals |
| 3 | `P3_Exp02_label_dependent_residual_extrapolation.m` | Export the residual model consumed by Parts 4 and 5 |
| 4 | `P4_Exp01_fixed_scale_LFP_optimization.m` | Main Fig. 2b–e and Main Fig. 3a–b |
| 4 | `P4_Exp02_fixed_scale_NMC_optimization.m` | Main Fig. 3c–d |
| 4 | `P4_Exp03_chemistry_comparison_summary_plots.m` | Main Fig. 3e–f |
| 5 | `P5_Exp01_scale_dependent_LFP_optimization.m` | Main Fig. 4a–f |
| 5 | `P5_Exp02_scale_dependent_NMC_optimization.m` | Main Fig. 5a–c |
| 5 | `P5_Exp03_NMC_error_scaling_scale_robustness.m` | Main Fig. 5d–f |
| 5 | `P5_Exp04_LFP_FixedKBenchmark_Heatmap.m` | Main Fig. 4g |
| 5 | `P5_Exp05_NMC_FixedKBenchmark_Heatmap.m` | Main Fig. 5g |

Two identical copies of `local_build_noise_by_k.m` are retained in the Part 4 and Part 5 `Function/` folders so that the existing relative paths remain unchanged.

## Requirements

- MATLAB
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox

The CNN training and nested-resampling experiments are computationally intensive. The principal training and resampling scripts use fixed random seeds where applicable.

## Full reproduction order

Run each script from its own Part directory.

```matlab
% Part 1
P1_Exp01_SourceDomain_QVPreprocessing
P1_Exp02_SourceDomain_CNNTraining_CV

% Part 2
P2_Exp01_target_data_processing
P2_Exp02_fine_tuning_L2SP_attention_prediction   % selected-method reference

% Part 3
P3_Exp01_NestedResampling_ErrorEvaluation
P3_Exp02_label_dependent_residual_extrapolation

% Part 4
P4_Exp01_fixed_scale_LFP_optimization
P4_Exp02_fixed_scale_NMC_optimization
P4_Exp03_chemistry_comparison_summary_plots

% Part 5
P5_Exp01_scale_dependent_LFP_optimization
P5_Exp02_scale_dependent_NMC_optimization
P5_Exp03_NMC_error_scaling_scale_robustness
P5_Exp04_LFP_FixedKBenchmark_Heatmap
P5_Exp05_NMC_FixedKBenchmark_Heatmap
```

Part 3 produces:

```text
Part3_Prediction error extrapolation/Output/Results/
    P3_Exp02_ResidualExtrapolation_Model.mat
```

This file is the direct prediction-to-TEA interface consumed by Parts 4 and 5.

## Inputs and generated results

Version-controlled inputs are located in:

- `Part1_Source-domain_Prediction/Source_Data/`
- `Part2_Target-domain_Prediction/Target_Data/`
- `Part4_Economic optimization under fixed scale/Input/`
- `Part5_Economic optimization under varied scale/Input/`

Generated `Output/` and `Figures/` directories are excluded by `.gitignore` and are recreated locally. Some retained core scripts also emit ancillary diagnostic or secondary outputs because those calculations are integrated with the main workflow; no standalone secondary-analysis scripts are retained.

## Citation

Citation information will be added when the accompanying manuscript is available.
