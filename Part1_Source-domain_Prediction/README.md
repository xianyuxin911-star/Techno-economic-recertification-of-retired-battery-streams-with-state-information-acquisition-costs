# Part 1 — Source-domain SOH Prediction

## Purpose

Part 1 constructs the source-domain battery dataset and trains the cell-level convolutional neural network (CNN) used by the downstream transfer-learning experiments.

The two scripts must be run in numerical order:

```text
Raw source-domain MAT files
        │
        ▼
P1_Exp01_SourceDomain_QVPreprocessing.m
        │
        ├── Processed source-domain dataset
        └── Supplementary Fig. 1
        │
        ▼
P1_Exp02_SourceDomain_CNNTraining_CV.m
        │
        ├── Source-domain CNN model
        ├── Cross-validation predictions and metrics
        └── Extended Data Fig. 1b
```

## Code-to-result map

### Step 1 — Preprocess the source-domain data

**Run**

```matlab
P1_Exp01_SourceDomain_QVPreprocessing
```

**Reads**

```text
Source_Data/MAP*.mat
```

**Produces**

```text
Output/Data/P1_SourceDomain_Dataset.mat
```

**Reproduces**

> **Supplementary Fig. 1a — Source-domain Q–V curves**  
> `Figures/Supplementary/SupFig01a_SourceDomain_QVCurves.png`

The script extracts Step-7 discharge Q–V curves over the normalized-capacity window 0.70–0.86, resamples each curve to 512 points, and constructs the source-domain CNN dataset.

---

### Step 2 — Train and evaluate the source-domain CNN

**Run**

```matlab
P1_Exp02_SourceDomain_CNNTraining_CV
```

**Reads**

```text
Output/Data/P1_SourceDomain_Dataset.mat
```

**Produces**

```text
Output/Models/P1_SourceDomain_CNN_Model.mat
Output/Results/P1_SourceDomain_CVPredictions.csv
Output/Results/P1_SourceDomain_CVMetrics.csv
```

**Reproduces**

> **Extended Data Fig. 1a — Source-domain prediction performance**  
> `Figures/Extended/ExD01a_SourceDomainPredictionPerformance.png`

The script performs five-fold cross-validation, exports sample- and fold-level prediction results, and trains the final source-domain CNN used by Parts 2 and 3.

Both figures are also saved as editable MATLAB `.fig` files in the same figure directories.

## Input data

### `Source_Data/`

This directory contains 96 source-domain battery MAT files:

```text
MAP150919190000001.mat ... MAP150919190000048.mat
MAP151007190000001.mat ... MAP151007190000048.mat
```

`P1_Exp01_SourceDomain_QVPreprocessing.m` searches this directory for files matching `MAP*.mat`.

## Script details

### 1. `P1_Exp01_SourceDomain_QVPreprocessing.m`

This script:

1. Loads each source-domain battery file from `Source_Data/`.
2. Locates the target discharge step (`Target_Step = 7`).
3. Extracts voltage as a function of normalized capacity.
4. Retains the normalized-capacity window from 0.70 to 0.86.
5. Interpolates every valid Q–V curve to 512 points.
6. Saves the processed dataset for CNN training.
7. Plots the source-domain Q–V curves coloured by SOH.

#### Generated dataset

```text
Output/Data/P1_SourceDomain_Dataset.mat
```

The MAT file contains:

| Variable | Description |
|---|---|
| `Train_X` | Resampled source-domain Q–V input features |
| `Train_Y` | Cell-level SOH labels |
| `SN_List` | Battery serial-number identifiers |
| `cfg` | Preprocessing configuration |

#### Manuscript figure

```text
Figures/Supplementary/SupFig01a_SourceDomain_QVCurves.png
```

**Corresponds to Supplementary Fig. 1a.**  
The figure displays the normalized-capacity Q–V curves of the source-domain cells, with curve colour indicating SOH.

### 2. `P1_Exp02_SourceDomain_CNNTraining_CV.m`

This script:

1. Loads `Output/Data/P1_SourceDomain_Dataset.mat`.
2. Performs five-fold cross-validation with a fixed random seed (`seed = 42`).
3. Trains the source-domain CNN and generates out-of-fold SOH predictions.
4. Calculates fold-level prediction metrics.
5. Trains the final source-domain CNN using all valid samples.
6. Saves the model used by Part 2 and Part 3.
7. Generates the source-domain prediction-performance figure.

#### Generated model

```text
Output/Models/P1_SourceDomain_CNN_Model.mat
```

The saved model file contains the final network, normalization parameters, configuration, fold metrics, labels, and out-of-fold predictions. Part 2 loads this model for target-domain transfer learning.

#### Generated numerical results

```text
Output/Results/P1_SourceDomain_CVPredictions.csv
Output/Results/P1_SourceDomain_CVMetrics.csv
```

| Result file | Description |
|---|---|
| `P1_SourceDomain_CVPredictions.csv` | Sample-level true SOH, cross-validated predicted SOH, and prediction error |
| `P1_SourceDomain_CVMetrics.csv` | Prediction metrics for the five cross-validation folds |

#### Manuscript figure

```text
Figures/Extended/ExD01a_SourceDomainPredictionPerformance.png
```

**Corresponds to Extended Data Fig. 1a.**  
The figure combines:

- a true-versus-predicted SOH parity plot;
- the distribution of true SOH;
- the cross-validation prediction-error distribution.

## How to run

Open MATLAB, change the current folder to `Part1_Source-domain_Prediction`, and run:

```matlab
P1_Exp01_SourceDomain_QVPreprocessing
P1_Exp02_SourceDomain_CNNTraining_CV
```

The scripts automatically create the required `Output/` and `Figures/` subdirectories.

## Dependencies and downstream use

- `P1_Exp02_SourceDomain_CNNTraining_CV.m` requires the dataset generated by `P1_Exp01_SourceDomain_QVPreprocessing.m`.
- Part 2 loads `Output/Models/P1_SourceDomain_CNN_Model.mat` for target-domain prediction and transfer learning.
- Part 3 also uses the source-domain model during nested-resampling experiments.
- MATLAB Deep Learning Toolbox is required for CNN training.
- Statistics and Machine Learning Toolbox is used for statistical evaluation and density estimation.

## Repository policy

The source MATLAB scripts and raw battery data are version-controlled. Generated `Output/` and `Figures/` directories remain local and are excluded from GitHub through the root `.gitignore`.
