%% P2_Exp03_frozen_encoder_attention_prediction.m
% Purpose:
% This script performs target-domain SOH prediction using a frozen
% source-domain CNN encoder. Cell-level embeddings are extracted from target
% pack voltage curves using the frozen source-domain encoder. A soft-min
% attention module and an MLP regression head are trained on target-domain
% pack labels and evaluated using leave-one-pack-out cross-validation.

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
    'experiment_id', 'P2_Exp03', ...
    'method', 'FrozenEncoder_Attention_MLP', ...
    'target_mat', fullfile(part_dir, 'Output', 'Data', 'P2_Exp01_TargetPack_Input.mat'), ...
    'source_mat', fullfile(part_dir, '..', 'Part1_Source-domain_Prediction', 'Output', 'Models', 'P1_SourceDomain_CNN_Model.mat'), ...
    'pred_csv', fullfile(output_results_dir, 'P2_Exp03_FrozenEncoder_PackPredictions.csv'), ...
    'metrics_csv', fullfile(output_results_dir, 'P2_Exp03_FrozenEncoder_Metrics.csv'), ...
    'result_mat', fullfile(output_results_dir, 'P2_Exp03_FrozenEncoder_Results.mat'), ...
    'embed_layer', 'gap', ...
    'force_recompute_embed', true, ...
    'use_source_norm', true, ...
    'attn_hidden', 32, ...
    'attn_tau', 0.10, ...
    'head_hidden', 16, ...
    'epochs', 400, ...
    'lr', 1e-3, ...
    'l2', 1e-4, ...
    'seed', 42);

rng(cfg.seed);

%% 1) Load target-domain pack dataset
if ~exist(cfg.target_mat, 'file')
error('Target-domain input file not found: %s. Please run P2_Exp01_target_data_processing.m first.', cfg.target_mat);
end

T = load(cfg.target_mat);

X_img = T.X_img;
y_all = double(T.y_all(:));

if isfield(T, 'SN_List')
    SN_List = T.SN_List(:);
else
    SN_List = strings(numel(y_all), 1);
    for i = 1:numel(y_all)
        SN_List(i) = sprintf('Pack_%d', i);
    end
end

if isfield(T, 'cfg')
    cfgT = T.cfg;
else
    cfgT = struct();
end

[nCell, M, ~, Npack] = size(X_img);

fprintf('[Target] Dataset loaded | Npack = %d | M = %d | Cells = %d\n', ...
    Npack, M, nCell);

fprintf('[Target] SOH range: %.4f - %.4f\n', min(y_all), max(y_all));

if isfield(cfgT, 'qmin') && isfield(cfgT, 'qmax')
    fprintf('[Target] Partial window: [%.2f, %.2f]\n', cfgT.qmin, cfgT.qmax);
end

%% 2) Load source-domain model
if ~exist(cfg.source_mat, 'file')
    error('Source-domain model not found: %s. Please check Part1 output.', cfg.source_mat);
end

S = load(cfg.source_mat);

netFull  = S.netFull;
muX_full = double(S.muX_full);
sgX_full = double(S.sgX_full);

if sgX_full < 1e-12
    sgX_full = 1;
end

if isfield(S, 'cfg')
    cfgS = S.cfg;
else
    cfgS = struct();
end

fprintf('[Source] Model loaded successfully.\n');

if isfield(cfgS, 'qmin') && isfield(cfgS, 'qmax') && isfield(cfgS, 'M')
    fprintf('[Source] Training window: [%.2f, %.2f] | Input length M = %d\n', ...
        cfgS.qmin, cfgS.qmax, cfgS.M);
end

%% 3) Safety checks
% Check input length consistency.
if isfield(cfgS, 'M')
    if M ~= cfgS.M
        error('Input-length mismatch: target-domain M = %d, but source model requires M = %d.', ...
            M, cfgS.M);
    end
end

% Check whether source and target domains use the same partial-discharge window.
if isfield(cfgT, 'qmin') && isfield(cfgT, 'qmax') && ...
        isfield(cfgS, 'qmin') && isfield(cfgS, 'qmax')

    if abs(cfgT.qmin - cfgS.qmin) > 1e-3 || ...
            abs(cfgT.qmax - cfgS.qmax) > 1e-3

        warning(['Source and target qmin/qmax settings are inconsistent.\n' ...
            'Source: [%.2f, %.2f]\n' ...
            'Target: [%.2f, %.2f]\n' ...
            'This may lead to unreliable prediction results.'], ...
            cfgS.qmin, cfgS.qmax, cfgT.qmin, cfgT.qmax);
    end
else
    warning('Window consistency cannot be verified. Please manually check the source and target files.');
end

%% 4) Pre-compute cell-level embeddings
fprintf('\n[Embedding] Extracting cell-level embeddings using source-domain normalization...\n');

Z_all = cell(Npack, 1);
D = [];

for p = 1:Npack

    % Extract all cell curves from one pack: 45 x M.
    Vrs = squeeze(X_img(:, :, 1, p));

    % Normalize target-domain curves using source-domain statistics.
    if cfg.use_source_norm
        VrsN = (double(Vrs) - muX_full) / sgX_full;
    else
        VrsN = double(Vrs);
    end

    VrsN = single(VrsN);

    % Reshape input for the CNN encoder.
    X4 = reshape(VrsN', [1 M 1 nCell]);

    % Extract embeddings from the frozen CNN encoder.
    Z = local_get_embeddings(netFull, X4, cfg.embed_layer);

    if isempty(D)
        D = size(Z, 1);
    end

    Z_all{p} = single(Z);
end

fprintf('[Embedding] Feature extraction completed. Embedding dimension D = %d\n', D);

%% 5) Leave-one-pack-out cross-validation
y_pred = nan(Npack, 1);
MaxAttn_List = nan(Npack, 1);

fprintf('\nStarting leave-one-pack-out cross-validation...\n');

for k = 1:Npack

    % Prepare training and test packs.
    testPack = k;
    trainPacks = setdiff(1:Npack, testPack);

    y_tr = y_all(trainPacks);
    Z_tr = Z_all(trainPacks);

    % Standardize labels within the current fold.
    muY = mean(y_tr);
    sgY = std(y_tr);

    if sgY < 1e-12
        sgY = 1;
    end

    y_trN = (y_tr - muY) / sgY;

    % Initialize attention and regression-head parameters.
    params = local_init_params(D, cfg);
    avgG   = local_zeros_like(params);
    avgSqG = local_zeros_like(params);

    iter = 0;
    nTrain = numel(Z_tr);

    % Training loop.
    for ep = 1:cfg.epochs
        order = randperm(nTrain);

        for ii = 1:nTrain
            iter = iter + 1;

            Zi = dlarray(single(Z_tr{order(ii)}));
            yi = dlarray(single(y_trN(order(ii))));

            [loss, grads] = dlfeval(@local_grad_one, params, Zi, yi, cfg);

            [params, avgG, avgSqG] = local_adam_step( ...
                params, grads, avgG, avgSqG, iter, cfg.lr);
        end
    end

    % Test on the held-out pack.
    Zte = dlarray(single(Z_all{testPack}));
    [yhatN, attn] = local_forward(params, Zte, cfg);

    % Transform prediction back to the original SOH scale.
    y_pred(testPack) = double(gather(extractdata(yhatN))) * sgY + muY;

    % Record the maximum attention score.
    MaxAttn_List(testPack) = max(double(gather(extractdata(attn))));

    fprintf('[Fold %d/%d] True=%.4f | Pred=%.4f | Error=%.4f | MaxAttn=%.2f\n', ...
        k, Npack, y_all(testPack), y_pred(testPack), ...
        y_pred(testPack) - y_all(testPack), MaxAttn_List(testPack));
end

%% 6) Evaluate prediction performance
err = y_pred - y_all;

RMSE = sqrt(mean(err.^2));
MAE  = mean(abs(err));
MAPE = mean(abs(err ./ y_all)) * 100;

sst = sum((y_all - mean(y_all)).^2);
sse = sum(err.^2);
R2  = 1 - sse / sst;

fprintf('\n========== Final frozen-encoder results ==========\n');

if isfield(cfgT, 'qmin') && isfield(cfgT, 'qmax')
    fprintf('Partial window = %.2f - %.2f\n', cfgT.qmin, cfgT.qmax);
end

fprintf('R2   = %.4f\n', R2);
fprintf('RMSE = %.4f\n', RMSE);
fprintf('MAE  = %.4f\n', MAE);
fprintf('MAPE = %.4f%%\n', MAPE);

%% 7) Export prediction results
fprintf('\nExporting frozen-encoder prediction results...\n');

% Pack-level prediction results.
Export_SOH = table( ...
    (1:Npack)', ...
    SN_List, ...
    y_all, ...
    y_pred, ...
    err, ...
    abs(err), ...
    MaxAttn_List, ...
    'VariableNames', { ...
    'Pack_Index', ...
    'Pack_ID', ...
    'True_SOH', ...
    'Predicted_SOH', ...
    'Error', ...
    'Abs_Error', ...
    'Max_Attention_Score'});

% Model-level performance metrics.
Export_Metrics = table( ...
    {'Frozen_Encoder_Attention_MLP'}, ...
    R2, ...
    RMSE, ...
    MAE, ...
    MAPE, ...
    'VariableNames', { ...
    'Model_Type', ...
    'R2', ...
    'RMSE', ...
    'MAE', ...
    'MAPE_percent'});

% Write CSV files.
writetable(Export_SOH, cfg.pred_csv);
writetable(Export_Metrics, cfg.metrics_csv);

% Save complete MATLAB result file for downstream analysis.
save(cfg.result_mat, ...
    'cfg', ...
    'cfgT', ...
    'cfgS', ...
    'SN_List', ...
    'X_img', ...
    'y_all', ...
    'y_pred', ...
    'err', ...
    'RMSE', ...
    'MAE', ...
    'MAPE', ...
    'R2', ...
    'MaxAttn_List', ...
    'Z_all', ...
    'Export_SOH', ...
    'Export_Metrics');

fprintf('[Export] Export completed.\n');
fprintf('  - Pack-level predictions: %s\n', cfg.pred_csv);
fprintf('  - Performance metrics:    %s\n', cfg.metrics_csv);
fprintf('  - MATLAB result file:     %s\n', cfg.result_mat);


%% Local functions
function Z = local_get_embeddings(net, X4, layerName)
    A = activations(net, X4, layerName);

    if ndims(A) == 4
        Z = reshape(A, size(A, 3), size(A, 4));
    else
        Z = A;
    end

    Z = single(Z);
end

function params = local_init_params(D, cfg)
    scale = sqrt(2 / D);

    params.W1 = dlarray(scale * randn(cfg.attn_hidden, D, 'single'));
    params.b1 = dlarray(zeros(cfg.attn_hidden, 1, 'single'));
    params.w2 = dlarray(scale * randn(1, cfg.attn_hidden, 'single'));
    params.b2 = dlarray(zeros(1, 1, 'single'));

    params.Wr1 = dlarray(scale * randn(cfg.head_hidden, D, 'single'));
    params.br1 = dlarray(zeros(cfg.head_hidden, 1, 'single'));
    params.Wr2 = dlarray(scale * randn(1, cfg.head_hidden, 'single'));
    params.br2 = dlarray(zeros(1, 1, 'single'));
end

function [yhat, attn] = local_forward(params, Z, cfg)
    % 1. Attention module with soft-min behavior.
    H = relu(params.W1 * Z + params.b1);
    score = -(params.w2 * H + params.b2) / cfg.attn_tau;
    attn = local_softmax(score, 2);

    % 2. Cell-to-pack embedding aggregation.
    z_pack = Z * attn.';

    % 3. MLP regression head.
    h = relu(params.Wr1 * z_pack + params.br1);
    yhat = params.Wr2 * h + params.br2;
end

function y = local_softmax(x, dim)
    x = x - max(x, [], dim);
    ex = exp(x);
    y = ex ./ (sum(ex, dim) + 1e-12);
end

function [loss, grads] = local_grad_one(params, Z, y, cfg)
    [yhat, ~] = local_forward(params, Z, cfg);

    loss = (yhat - y).^2;

    if cfg.l2 > 0
        loss = loss + cfg.l2 * ( ...
            sum(params.W1(:).^2) + ...
            sum(params.w2(:).^2) + ...
            sum(params.Wr1(:).^2) + ...
            sum(params.Wr2(:).^2));
    end

    grads = dlgradient(loss, params);
end

function Zs = local_zeros_like(params)
    fn = fieldnames(params);

    for i = 1:numel(fn)
        Zs.(fn{i}) = dlarray(zeros(size(params.(fn{i})), ...
            'like', params.(fn{i})));
    end
end

function [params, avgG, avgSqG] = local_adam_step(params, grads, avgG, avgSqG, iter, lr)
    fn = fieldnames(params);

    for i = 1:numel(fn)
        [params.(fn{i}), avgG.(fn{i}), avgSqG.(fn{i})] = adamupdate( ...
            params.(fn{i}), ...
            grads.(fn{i}), ...
            avgG.(fn{i}), ...
            avgSqG.(fn{i}), ...
            iter, ...
            lr);
    end
end