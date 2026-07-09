%% P2_Exp06_regression_head_comparison.m
% Purpose:
% This script compares different regression heads for target-domain
% pack-level SOH prediction. A source-initialized encoder is fine-tuned with
% L2-SP regularization, and attention-pooled pack embeddings are extracted.
% Multiple regression heads, including MLP, Gaussian process regression, and
% support vector regression, are evaluated using leave-one-pack-out
% cross-validation.

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
    'experiment_id', 'P2_Exp06', ...
    'method', 'RegressionHead_Comparison', ...
    'target_mat', fullfile(part_dir, 'Output', 'Data', 'P2_Exp01_TargetPack_Input.mat'), ...
    'source_mat', fullfile(part_dir, '..', 'Part1_Source-domain_Prediction', 'Output', 'Models', 'P1_SourceDomain_CNN_Model.mat'), ...
    'pred_csv', fullfile(output_results_dir, 'P2_Exp06_RegressionHead_PackPredictions.csv'), ...
    'metrics_csv', fullfile(output_results_dir, 'P2_Exp06_RegressionHead_Metrics.csv'), ...
    'result_mat', fullfile(output_results_dir, 'P2_Exp06_RegressionHead_Results.mat'), ...
    'M', 512, ...
    'nCell', 45, ...
    'maxEpochs', 60, ...
    'warmupEpochs', 20, ...
    'lr_head', 1e-3, ...
    'lr_enc', 1e-5, ...
    'tune_last_conv', 3, ...
    'beta1', 0.9, ...
    'beta2', 0.999, ...
    'dEmbed', 64, ...
    'dAttH', 32, ...
    'dHeadH', 16, ...
    'attn_tau', 0.1, ...
    'l2_head', 1e-4, ...
    'lambda_sp', 0.005, ...
    'clip_enc', 1.0, ...
    'clip_head', 1.0, ...
    'seed', 42);

rng(cfg.seed);

fprintf('================ P2 Exp06: Regression-head comparison ================\n');
fprintf('Target-domain data file:  %s\n', cfg.target_mat);
fprintf('Source-domain model file: %s\n', cfg.source_mat);


%% 1) Load target-domain data
if ~exist(cfg.target_mat, 'file')
    error('Target-domain input file not found: %s. Please run P2_Exp01_target_data_processing.m first.', cfg.target_mat);
end

T = load(cfg.target_mat);

X_img = single(T.X_img);     % [45, M, 1, Npack]
y_all = single(T.y_all(:));

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

fprintf('[Target] Dataset loaded successfully | Npack = %d\n', size(X_img, 4));

%% 2) Load source-domain model and normalize target inputs
if ~exist(cfg.source_mat, 'file')
    error('Source-domain model not found: %s. Please check Part1 source-model output.', cfg.source_mat);
end

S = load(cfg.source_mat);

netFull = S.netFull;
muX = double(S.muX_full);
sgX = double(S.sgX_full);

if sgX < 1e-12
    sgX = 1;
end

% Normalize target-domain voltage curves using source-domain statistics.
X_img = (X_img - muX) / sgX;

Npack = size(X_img, 4);

%% 3) Prepare encoder and L2-SP reference model
layers = netFull.Layers;
idx_gap = find(arrayfun(@(L) strcmp(L.Name, 'gap'), layers), 1, 'first');
lgraphEnc = layerGraph(layers(1:idx_gap));

% L2-SP anchor.
enc_ref = dlnetwork(lgraphEnc);

fprintf('[INFO] Starting training and regression-head comparison...\n');

%% 4) Leave-one-pack-out training and regression-head comparison
yhat_MLP = nan(Npack, 1, 'single');
yhat_GPR = nan(Npack, 1, 'single');
yhat_SVM = nan(Npack, 1, 'single');

for te = 1:Npack

    tr = setdiff(1:Npack, te);

    % Standardize training labels within the current fold.
    y_train_raw = y_all(tr);
    muY = mean(y_train_raw);
    sgY = std(y_train_raw);

    if sgY < 1e-12
        sgY = 1;
    end

    y_train_norm = (y_train_raw - muY) / sgY;

    % Reset the trainable encoder for the current fold.
    currEnc = dlnetwork(lgraphEnc);

    % Initialize attention and MLP regression-head parameters.
    attW1 = dlarray(0.01 * randn(cfg.dAttH, cfg.dEmbed, 'single'));
    attb1 = dlarray(zeros(cfg.dAttH, 1, 'single'));
    attW2 = dlarray(0.01 * randn(1, cfg.dAttH, 'single'));
    attb2 = dlarray(zeros(1, 1, 'single'));

    headW1 = dlarray(0.01 * randn(cfg.dHeadH, cfg.dEmbed, 'single'));
    headb1 = dlarray(zeros(cfg.dHeadH, 1, 'single'));
    headW2 = dlarray(0.01 * randn(1, cfg.dHeadH, 'single'));
    headb2 = dlarray(zeros(1, 1, 'single'));

    % Adam optimizer states.
    avgG_enc = [];
    avgSqG_enc = [];
    avgG_head = cell(8, 1);
    avgSqG_head = cell(8, 1);

    iter = 0;

    % Stage 1: fine-tune encoder, attention module, and MLP head.
    for ep = 1:cfg.maxEpochs

        order = randperm(numel(tr));

        for ii = 1:numel(order)

            iter = iter + 1;
            idx_in_tr = order(ii);
            p = tr(idx_in_tr);

            raw_pack = X_img(:, :, 1, p);
            dlX = dlarray(reshape(raw_pack', [1 cfg.M 1 cfg.nCell]), 'SSCB');
            y_true = dlarray(y_train_norm(idx_in_tr));

            [~, grads] = dlfeval(@model_grads, dlX, y_true, currEnc, enc_ref, ...
                attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2, cfg);

            headParams = {attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2};
            headGrads  = {grads.attW1, grads.attb1, grads.attW2, grads.attb2, ...
                grads.headW1, grads.headb1, grads.headW2, grads.headb2};

            if cfg.clip_head > 0
                headGrads = local_clip_grads_cell(headGrads, cfg.clip_head);
            end

            if isempty(avgG_head{1})
                for k = 1:8
                    avgG_head{k} = [];
                    avgSqG_head{k} = [];
                end
            end

            for k = 1:8
                [headParams{k}, avgG_head{k}, avgSqG_head{k}] = adamupdate( ...
                    headParams{k}, headGrads{k}, avgG_head{k}, avgSqG_head{k}, ...
                    iter, cfg.lr_head, cfg.beta1, cfg.beta2);
            end

            [attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2] = headParams{:};

            if ep > cfg.warmupEpochs

                genc = grads.enc;
                genc = local_mask_enc_grads_keep_last_conv(currEnc, genc, cfg.tune_last_conv);

                if cfg.clip_enc > 0
                    genc = local_clip_grads_table(genc, cfg.clip_enc);
                end

                [currEnc, avgG_enc, avgSqG_enc] = adamupdate(currEnc, genc, ...
                    avgG_enc, avgSqG_enc, iter, cfg.lr_enc, cfg.beta1, cfg.beta2);
            end
        end
    end

    % Stage 2: extract attention-pooled pack embeddings.
    Feats_Train = zeros(numel(tr), cfg.dEmbed);
    Target_Train = double(y_train_norm);

    for i = 1:numel(tr)

        p = tr(i);
        raw_pack = X_img(:, :, 1, p);
        dlX = dlarray(reshape(raw_pack', [1 cfg.M 1 cfg.nCell]), 'SSCB');

        [zbag_val, ~] = extract_features_only( ...
            dlX, currEnc, attW1, attb1, attW2, attb2, cfg);

        Feats_Train(i, :) = double(extractdata(zbag_val))';
    end

    raw_te = X_img(:, :, 1, te);
    dlX_te = dlarray(reshape(raw_te', [1 cfg.M 1 cfg.nCell]), 'SSCB');

    [zbag_te, ypred_mlp_norm] = extract_features_only( ...
        dlX_te, currEnc, attW1, attb1, attW2, attb2, cfg, ...
        headW1, headb1, headW2, headb2);

    Feats_Test = double(extractdata(zbag_te))';

    % Stage 3: train and evaluate different regression heads.

    % 1. MLP head.
    val_mlp = double(extractdata(ypred_mlp_norm));
    yhat_MLP(te) = val_mlp * sgY + muY;

    % 2. Gaussian process regression head.
    try
        Mdl_GPR = fitrgp( ...
            Feats_Train, ...
            Target_Train, ...
            'KernelFunction', 'squaredexponential', ...
            'Standardize', false);

        pred_gpr_norm = predict(Mdl_GPR, Feats_Test);
        yhat_GPR(te) = pred_gpr_norm * sgY + muY;
    catch
        yhat_GPR(te) = yhat_MLP(te);
    end

    % 3. Support vector regression head.
    try
        Mdl_SVM = fitrsvm( ...
            Feats_Train, ...
            Target_Train, ...
            'KernelFunction', 'gaussian', ...
            'Standardize', false, ...
            'BoxConstraint', 10);

        pred_svm_norm = predict(Mdl_SVM, Feats_Test);
        yhat_SVM(te) = pred_svm_norm * sgY + muY;
    catch
        yhat_SVM(te) = yhat_MLP(te);
    end

    fprintf('[Fold %d] True = %.3f | MLP = %.3f, GPR = %.3f, SVM = %.3f\n', ...
        te, y_all(te), yhat_MLP(te), yhat_GPR(te), yhat_SVM(te));
end

%% 5) Evaluate and compare regression heads
Models = {'MLP', 'GPR', 'SVM'};
Preds  = {yhat_MLP, yhat_GPR, yhat_SVM};

fprintf('\n==================== Final model comparison ====================\n');
fprintf('%-10s | %-8s | %-8s | %-8s | %-8s\n', ...
    'Model', 'R2', 'RMSE', 'MAE', 'MAPE');
fprintf('--------------------------------------------------------------\n');

R2_all = [];
RMSE_all = [];
MAE_all = [];
MAPE_all = [];

for i = 1:numel(Models)

    nm = Models{i};
    yh = Preds{i};
    err = yh - y_all;

    RMSE = sqrt(mean(err.^2));
    MAE  = mean(abs(err));
    MAPE = mean(abs(err ./ y_all)) * 100;

    sst = sum((y_all - mean(y_all)).^2);
    sse = sum(err.^2);
    R2  = 1 - sse / sst;

    R2_all(end+1) = R2;
    RMSE_all(end+1) = RMSE;
    MAE_all(end+1) = MAE;
    MAPE_all(end+1) = MAPE;

    fprintf('%-10s | %.4f   | %.4f   | %.4f   | %.4f\n', ...
        nm, R2, RMSE, MAE, MAPE);
end

%% 6) Export multi-model prediction results
fprintf('\nExporting multi-model prediction results...\n');

% Pack-level SOH prediction results.
y_all_col    = double(y_all(:));
yhat_mlp_col = double(yhat_MLP(:));
yhat_gpr_col = double(yhat_GPR(:));
yhat_svm_col = double(yhat_SVM(:));

Export_SOH = table( ...
    (1:Npack)', ...
    SN_List, ...
    y_all_col, ...
    yhat_mlp_col, ...
    yhat_gpr_col, ...
    yhat_svm_col, ...
    'VariableNames', { ...
    'Pack_Index', ...
    'Pack_ID', ...
    'True_SOH', ...
    'Pred_MLP', ...
    'Pred_GPR', ...
    'Pred_SVM'});

% Model-level performance metrics.
Export_Metrics = table( ...
    Models', ...
    double(R2_all(:)), ...
    double(RMSE_all(:)), ...
    double(MAE_all(:)), ...
    double(MAPE_all(:)), ...
    'VariableNames', { ...
    'Model', ...
    'R2', ...
    'RMSE', ...
    'MAE', ...
    'MAPE'});

% Write CSV files.
writetable(Export_SOH, cfg.pred_csv);
writetable(Export_Metrics, cfg.metrics_csv);

% Save complete MATLAB result file for downstream analysis.
save(cfg.result_mat, ...
    'cfg', ...
    'cfgT', ...
    'SN_List', ...
    'y_all', ...
    'yhat_MLP', ...
    'yhat_GPR', ...
    'yhat_SVM', ...
    'Models', ...
    'Preds', ...
    'R2_all', ...
    'RMSE_all', ...
    'MAE_all', ...
    'MAPE_all', ...
    'Export_SOH', ...
    'Export_Metrics');

fprintf('[Export] Export completed.\n');
fprintf('  - Multi-model SOH predictions: %s\n', cfg.pred_csv);
fprintf('  - Multi-model performance metrics: %s\n', cfg.metrics_csv);
fprintf('  - MATLAB result file: %s\n', cfg.result_mat);

fprintf('\n================ P2 Exp06 completed ================\n');

%% Local helper functions

function [loss, grads] = model_grads(X, y, net, net_ref, attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2, cfg)
% Compute loss and gradients for L2-SP fine-tuning with the MLP head.

    [Z_raw, ~] = predict(net, X);
    Z = stripdims(squeeze(Z_raw));

    % Forward MLP.
    H = relu(attW1 * Z + attb1);
    score = -(attW2 * H + attb2) / cfg.attn_tau;
    score = score - max(score, [], 2);
    w = exp(score) ./ sum(exp(score), 2);
    zbag = Z * w.';

    h1 = relu(headW1 * zbag + headb1);
    ypred = headW2 * h1 + headb2;

    % Losses.
    mse_loss = mean((ypred - y).^2);

    l2_head_term = sum(attW1(:).^2) + ...
                   sum(attW2(:).^2) + ...
                   sum(headW1(:).^2) + ...
                   sum(headW2(:).^2);

    l2_sp_term = 0;

    if cfg.lambda_sp > 0

        T_curr = net.Learnables;
        T_ref  = net_ref.Learnables;

        for i = 1:height(T_curr)
            w_c = T_curr.Value{i};
            w_r = T_ref.Value{i};
            diff = w_c - w_r;
            l2_sp_term = l2_sp_term + sum(diff(:).^2);
        end
    end

    loss = mse_loss + cfg.l2_head * l2_head_term + cfg.lambda_sp * l2_sp_term;

    all_grads = dlgradient(loss, ...
        {net.Learnables, attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2});

    grads.enc    = all_grads{1};
    grads.attW1  = all_grads{2};
    grads.attb1  = all_grads{3};
    grads.attW2  = all_grads{4};
    grads.attb2  = all_grads{5};
    grads.headW1 = all_grads{6};
    grads.headb1 = all_grads{7};
    grads.headW2 = all_grads{8};
    grads.headb2 = all_grads{9};
end

function [zbag, ypred] = extract_features_only(X, net, attW1, attb1, attW2, attb2, cfg, headW1, headb1, headW2, headb2)

    [Z_raw, ~] = predict(net, X);
    Z = stripdims(squeeze(Z_raw));

    H = relu(attW1 * Z + attb1);
    score = -(attW2 * H + attb2) / cfg.attn_tau;
    score = score - max(score, [], 2);
    w = exp(score) ./ sum(exp(score), 2);

    zbag = Z * w.';

    if nargin > 7
        h1 = relu(headW1 * zbag + headb1);
        ypred = headW2 * h1 + headb2;
    else
        ypred = [];
    end
end

function genc = local_mask_enc_grads_keep_last_conv(net, genc, keep_last_conv)

    L = net.Layers;
    layerNames = string({L.Name});
    isConv = false(size(L));

    for i = 1:numel(L)

        cls = class(L(i));

        if contains(cls, "Convolution", "IgnoreCase", true) || ...
                contains(L(i).Name, "conv", "IgnoreCase", true)
            isConv(i) = true;
        end
    end

    convIdx = find(isConv);

    if isempty(convIdx)
        learnLayer = string(genc.Layer);
        uniq = unique(learnLayer, 'stable');
        keep = uniq(max(1, end-4):end);
    else
        k = min(keep_last_conv, numel(convIdx));
        keepConvNames = layerNames(convIdx(end-k+1:end));
        keep = keepConvNames;
    end

    for i = 1:height(genc)

        if ~ismember(string(genc.Layer(i)), keep)

            gi = genc.Value{i};

            if isempty(gi)
                continue;
            end

            gi_data = extractdata(gi);
            z = zeros(size(gi_data), 'like', gi_data);
            genc.Value{i} = dlarray(z);
        end
    end
end

function gtab = local_clip_grads_table(gtab, clipnorm)

    ss = 0;

    for i = 1:height(gtab)

        gi = gtab.Value{i};

        if ~isempty(gi)
            gi_data = extractdata(gi);
            ss = ss + sum(gi_data(:).^2);
        end
    end

    gn = sqrt(ss + 1e-12);

    if gn > clipnorm

        scale = clipnorm / gn;

        for i = 1:height(gtab)

            gi = gtab.Value{i};

            if ~isempty(gi)
                gtab.Value{i} = gi * scale;
            end
        end
    end
end

function gcell = local_clip_grads_cell(gcell, clipnorm)

    ss = 0;

    for i = 1:numel(gcell)

        gi = gcell{i};

        if ~isempty(gi)
            gi_data = extractdata(gi);
            ss = ss + sum(gi_data(:).^2);
        end
    end

    gn = sqrt(ss + 1e-12);

    if gn > clipnorm

        scale = clipnorm / gn;

        for i = 1:numel(gcell)

            if ~isempty(gcell{i})
                gcell{i} = gcell{i} * scale;
            end
        end
    end
end