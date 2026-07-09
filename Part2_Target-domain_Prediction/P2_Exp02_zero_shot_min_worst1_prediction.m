%% P2_Exp02_zero_shot_min_worst1_prediction.m
% Purpose: This script performs zero-shot target-domain SOH prediction using the
% source-domain CNN model without target-domain fine-tuning. For each target
% pack, the source model predicts SOH values for 45 cell-voltage curves, and
% the minimum predicted cell-level SOH is used as the pack-level SOH
% prediction under the Min/Worst-1 aggregation strategy.

clc; clear; close all;

%% Configuration
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end

part_dir = script_dir;

output_results_dir = fullfile(part_dir, 'Output', 'Results');
if ~exist(output_results_dir, 'dir')
    mkdir(output_results_dir);
end

cfg = struct( ...
    'experiment_id', 'P2_Exp02', ...
    'method', 'ZeroShot_MinWorst1', ...
    'aggregation', 'Min_Worst_1', ...
    'target_mat', fullfile(part_dir, 'Output', 'Data', 'P2_Exp01_TargetPack_Input.mat'), ...
    'source_mat', fullfile(part_dir, '..', 'Part1_Source-domain_Prediction', 'Output', 'Models', 'P1_SourceDomain_CNN_Model.mat'), ...
    'metrics_csv', fullfile(output_results_dir, 'P2_Exp02_ZeroShot_MinWorst1_Metrics.csv'), ...
    'pack_csv', fullfile(output_results_dir, 'P2_Exp02_ZeroShot_MinWorst1_PackPredictions.csv'), ...
    'cell_csv', fullfile(output_results_dir, 'P2_Exp02_ZeroShot_MinWorst1_CellPredictions.csv'), ...
    'result_mat', fullfile(output_results_dir, 'P2_Exp02_ZeroShot_MinWorst1_Results.mat'));

%% 1) Load source model and target-domain data
if ~exist(cfg.target_mat, 'file')
    error('Target-domain dataset not found: %s', cfg.target_mat);
end

if ~exist(cfg.source_mat, 'file')
    error('Source-domain model not found: %s', cfg.source_mat);
end

T = load(cfg.target_mat);
S = load(cfg.source_mat);

X_img = T.X_img;

% Convert labels to a column vector to avoid dimension mismatch.
y_true = double(T.y_all(:));

[nCell, M, ~, Npack] = size(X_img);

netFull  = S.netFull;
muX_full = double(S.muX_full);
sgX_full = double(S.sgX_full);
muY_full = double(S.muY_full);
sgY_full = double(S.sgY_full);

fprintf('[INFO] Starting zero-shot target-domain inference...\n');
fprintf('[INFO] Number of target packs: %d\n', Npack);
fprintf('[INFO] Number of cells per pack: %d\n', nCell);
fprintf('[INFO] Feature length: %d\n', M);

%% 2) Zero-shot inference with Min/Worst-1 aggregation
Pred_MinWorst1 = zeros(Npack, 1);
Worst_Cell_Index = zeros(Npack, 1);
All_Cell_Preds = zeros(nCell, Npack);

fprintf('\nRunning inference...\n');

for p = 1:Npack

    Vrs = squeeze(X_img(:, :, 1, p)); % 45 x M

    % Standardize voltage curves and reshape them for CNN inference.
    VrsN = (double(Vrs) - muX_full) / sgX_full;
    X_input = reshape(VrsN', [1, M, 1, nCell]);

    % Predict SOH for 45 cells.
    y_hat_norm = predict(netFull, X_input, 'ExecutionEnvironment', 'auto');
    y_hat_abs  = double(y_hat_norm) * sgY_full + muY_full;
    y_hat_abs  = y_hat_abs(:);

    % Save cell-level predictions.
    All_Cell_Preds(:, p) = y_hat_abs;

    % Min/Worst-1 aggregation.
    [Pred_MinWorst1(p), Worst_Cell_Index(p)] = min(y_hat_abs);

    if mod(p, 5) == 0 || p == Npack
        fprintf('  [Progress] Completed %d / %d packs.\n', p, Npack);
    end
end

%% 3) Evaluate Min/Worst-1 zero-shot prediction
fprintf('\n======================================================\n');
fprintf('              Zero-shot Min/Worst-1 result            \n');
fprintf('======================================================\n');

% Metrics: RMSE, MAE, MAPE, and R2.
RMSE = sqrt(mean((y_true - Pred_MinWorst1).^2));
MAE  = mean(abs(y_true - Pred_MinWorst1));
MAPE = mean(abs((y_true - Pred_MinWorst1) ./ y_true)) * 100;
R2   = 1 - sum((y_true - Pred_MinWorst1).^2) / ...
    sum((y_true - mean(y_true)).^2);

fprintf('%-15s | RMSE    | MAE     | MAPE     | R2\n', 'Strategy');
fprintf('--------------------------------------------------------\n');
fprintf('%-15s | %.5f | %.5f | %5.2f%%   | %.4f\n', ...
    'Min/Worst-1', RMSE, MAE, MAPE, R2);
fprintf('--------------------------------------------------------\n');

Metrics_Summary = table( ...
    {'Min_Worst_1'}, ...
    RMSE, ...
    MAE, ...
    MAPE, ...
    R2, ...
    'VariableNames', {'Strategy', 'RMSE', 'MAE', 'MAPE', 'R2'});

%% 4) Export zero-shot Min/Worst-1 results
fprintf('\nExporting zero-shot Min/Worst-1 results...\n');

% Export strategy-level metrics.
writetable(Metrics_Summary, cfg.metrics_csv);

% Export pack-level Min/Worst-1 predictions.
ExportPack = table( ...
    (1:Npack)', ...
    y_true, ...
    Pred_MinWorst1, ...
    Worst_Cell_Index, ...
    'VariableNames', { ...
    'Pack_Index', ...
    'True_Pack_SOH', ...
    'Pred_MinWorst1_SOH', ...
    'Worst_Cell_Index'});

writetable(ExportPack, cfg.pack_csv);

% Export cell-level predictions for all packs.
cell_col_names = arrayfun(@(x) sprintf('Pack_%d', x), ...
    1:Npack, 'UniformOutput', false);

Cell_Table = array2table(All_Cell_Preds, ...
    'VariableNames', cell_col_names);

Cell_Table.Cell_Index = (1:nCell)';

Cell_Table = movevars(Cell_Table, 'Cell_Index', 'Before', 1);

writetable(Cell_Table, cfg.cell_csv);

% Save the complete MATLAB result file for downstream scripts.
save(cfg.result_mat, ...
    'cfg', ...
    'y_true', ...
    'Pred_MinWorst1', ...
    'Worst_Cell_Index', ...
    'All_Cell_Preds', ...
    'Metrics_Summary', ...
    'ExportPack', ...
    'Cell_Table');

fprintf('[EXPORT] Export completed.\n');
fprintf('  - Min/Worst-1 metrics:          %s\n', cfg.metrics_csv);
fprintf('  - Min/Worst-1 pack predictions: %s\n', cfg.pack_csv);
fprintf('  - Cell-level predictions:       %s\n', cfg.cell_csv);
fprintf('  - MATLAB result file:           %s\n', cfg.result_mat);
