# Techno-economic recertification of retired battery streams with state-information acquisition costs

This repository provides the MATLAB implementation and battery input data for a cell-to-pack transfer-learning and techno-economic analysis framework for retired battery recertification.

The workflow connects battery state-of-health (SOH) prediction with downstream recertification decisions. It first trains a source-domain cell-level model, transfers the learned representation to target-domain battery packs, evaluates how prediction uncertainty changes with the number of labelled packs, and then optimizes state-information acquisition and recertification decisions under fixed and varying deployment scales.

## Workflow

The project is organized into five sequential parts:

1. **Source-domain prediction**  
   Preprocess source-domain battery data and train a cell-level convolutional neural network for SOH prediction.

2. **Target-domain prediction**  
   Transfer the source-domain representation to pack-level SOH prediction and compare zero-shot, frozen-encoder, fine-tuning, L2-SP, regression-head, and attention-pooling strategies.

3. **Prediction-error extrapolation**  
   Use nested resampling to quantify label-dependent prediction errors and extrapolate the residual-error distribution as the number of labelled target packs changes.

4. **Economic optimization under fixed scale**  
   Optimize the number of labelled packs and SOH decision thresholds for LFP and NMC battery recertification under a fixed deployment scale.

5. **Economic optimization under varied scale**  
   Extend the optimization to scale-dependent deployment, robustness analyses, fixed-label benchmarks, and market-scale scenarios.

## Repository structure

```text
.
├── Part1_Source-domain_Prediction/
│   ├── Source_Data/                 # Source-domain battery input data
│   ├── P1_Exp01_*.m                 # Data preprocessing
│   └── P1_Exp02_*.m                 # CNN training and cross-validation
├── Part2_Target-domain_Prediction/
│   ├── Target_Data/                 # Target-domain battery-pack input data
│   └── P2_Exp01_*.m ... P2_Exp11_*.m
├── Part3_Prediction error extrapolation/
│   └── P3_Exp01_*.m ... P3_Exp04_*.m
├── Part4_Economic optimization under fixed scale/
│   ├── Input/                       # LFP and NMC SOH input tables
│   ├── Function/                    # Shared helper function
│   └── P4_Exp01_*.m ... P4_Exp10_*.m
└── Part5_Economic optimization under varied scale/
    ├── Input/                       # LFP and NMC SOH input tables
    ├── Function/                    # Shared helper function
    └── P5_Exp01_*.m ... P5_Exp07_*.m
```

Generated `Output/` and `Figures/` directories are intentionally excluded from the repository. They are recreated locally when the corresponding scripts are run.

## Requirements

- MATLAB
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox

The deep-learning experiments can be computationally intensive. Random seeds are fixed in the main training and resampling scripts where applicable.

## Reproduction workflow

Run scripts from their own Part directory. The scripts determine paths relative to their file locations.

### Part 1: source-domain model

```matlab
P1_Exp01_SourceDomain_QVPreprocessing
P1_Exp02_SourceDomain_CNNTraining_CV
```

These scripts preprocess the source-domain battery files and generate the source CNN model required by Parts 2 and 3.

### Part 2: target-domain transfer learning

First run:

```matlab
P2_Exp01_target_data_processing
```

Then run the required prediction experiments:

```matlab
P2_Exp02_zero_shot_min_worst1_prediction
P2_Exp03_frozen_encoder_attention_prediction
P2_Exp04_fine_tuning_attention_prediction
P2_Exp05_fine_tuning_L2SP_attention_prediction
```

The remaining Part 2 scripts perform regression-head comparisons, attention-pooling ablations, transfer-strategy summaries, and fine-tuning-depth analyses.

### Part 3: label-dependent error model

```matlab
P3_Exp01_NestedResampling_ErrorEvaluation
P3_Exp02_label_dependent_error_summary
P3_Exp03_label_dependent_residual_extrapolation
P3_Exp04_ResidualRegionBoundaryRobustness
```

Part 3 depends on the processed target data and source model produced by Parts 1 and 2.

### Part 4: fixed-scale optimization

Run `P4_Exp01_fixed_scale_LFP_optimization.m` and `P4_Exp02_fixed_scale_NMC_optimization.m` first. The subsequent Part 4 scripts use these baseline results for chemistry comparisons, regret benchmarks, and sensitivity or robustness analyses.

### Part 5: scale-dependent optimization

Run `P5_Exp01_scale_dependent_LFP_optimization.m` followed by `P5_Exp02_scale_dependent_NMC_optimization.m`. The remaining scripts provide robustness analyses, fixed-label benchmarks, per-pack comparisons, and the extended market-scale analysis.

## Data and results

The repository contains the battery input data needed by the MATLAB workflow:

- `Part1_Source-domain_Prediction/Source_Data/`
- `Part2_Target-domain_Prediction/Target_Data/`
- `Part4_Economic optimization under fixed scale/Input/`
- `Part5_Economic optimization under varied scale/Input/`

Intermediate models, generated workspaces, numerical results, and figures are not version-controlled. They are saved locally under the relevant `Output/` and `Figures/` directories.

## Citation

Citation information will be added when the accompanying manuscript is available.

