# Part 1 — Source-domain SOH model

## Core role

Part 1 prepares the source-domain battery dataset and trains the cell-level CNN that initializes the downstream target-domain and nested-resampling calculations.

## Run order

```matlab
P1_Exp01_SourceDomain_QVPreprocessing
P1_Exp02_SourceDomain_CNNTraining_CV
```

### `P1_Exp01_SourceDomain_QVPreprocessing.m`

Reads:

```text
Source_Data/MAP*.mat
```

Produces:

```text
Output/Data/P1_SourceDomain_Dataset.mat
```

The script extracts and resamples the source-domain Q–V curves used for CNN training.

### `P1_Exp02_SourceDomain_CNNTraining_CV.m`

Reads:

```text
Output/Data/P1_SourceDomain_Dataset.mat
```

Produces:

```text
Output/Models/P1_SourceDomain_CNN_Model.mat
Output/Results/P1_SourceDomain_CVPredictions.csv
Output/Results/P1_SourceDomain_CVMetrics.csv
```

`P1_SourceDomain_CNN_Model.mat` is the required source-model input for `P2_Exp05` and `P3_Exp01`.

## Dependencies

- MATLAB Deep Learning Toolbox
- MATLAB Statistics and Machine Learning Toolbox

Generated `Output/` and `Figures/` directories are excluded from version control.
