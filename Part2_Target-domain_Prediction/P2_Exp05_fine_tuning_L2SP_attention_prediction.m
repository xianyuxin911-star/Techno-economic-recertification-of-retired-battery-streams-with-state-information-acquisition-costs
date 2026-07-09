%% P2_Exp05_fine_tuning_L2SP_attention_prediction.m
% Purpose:
% This script performs target-domain SOH prediction using source-initialized
% encoder fine-tuning with L2-SP regularization. The encoder is initialized
% from the source-domain CNN model and constrained toward the source-domain
% starting point during fine-tuning. A soft-min attention module and an MLP
% regression head are trained with leave-one-pack-out cross-validation.

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
    'experiment_id', 'P2_Exp05', ...
    'method', 'FineTune_L2SP_Attention_MLP', ...
    'target_mat', fullfile(part_dir, 'Output', 'Data', 'P2_Exp01_TargetPack_Input.mat'), ...
    'source_mat', fullfile(part_dir, '..', 'Part1_Source-domain_Prediction', 'Output', 'Models', 'P1_SourceDomain_CNN_Model.mat'), ...
    'pred_csv', fullfile(output_results_dir, 'P2_Exp05_FineTuneL2SP_PackPredictions.csv'), ...
    'attn_csv', fullfile(output_results_dir, 'P2_Exp05_FineTuneL2SP_CellAttention.csv'), ...
    'metrics_csv', fullfile(output_results_dir, 'P2_Exp05_FineTuneL2SP_Metrics.csv'), ...
    'result_mat', fullfile(output_results_dir, 'P2_Exp05_FineTuneL2SP_Results.mat'), ...
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

fprintf('================ P2 Exp05: Fine-tuning with L2-SP ================\n');
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

enc_ref = dlnetwork(lgraphEnc);

fprintf('[INFO] Starting fine-tuning with L2-SP regularization...\n');
fprintf('       Head L2: %.1e | Encoder L2-SP: %.1e\n', cfg.l2_head, cfg.lambda_sp);

%% 4) Leave-one-pack-out fine-tuning with L2-SP
yhat = nan(Npack, 1, 'single');
MaxAttn_List = nan(Npack, 1);
Cell_Attn_All = zeros(cfg.nCell, Npack);

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

    % Initialize attention and regression-head parameters.
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

    % Training loop.
    for ep = 1:cfg.maxEpochs

        order = randperm(numel(tr));

        for ii = 1:numel(order)

            iter = iter + 1;
            idx_in_tr = order(ii);
            p = tr(idx_in_tr);

            % Construct one-pack mini-batch.
            raw_pack = X_img(:, :, 1, p);
            dlX = dlarray(reshape(raw_pack', [1 cfg.M 1 cfg.nCell]), 'SSCB');
            y_true = dlarray(y_train_norm(idx_in_tr));

            % Compute gradients with L2-SP regularization.
            [loss, grads] = dlfeval(@model_grads, dlX, y_true, currEnc, enc_ref, ...
                attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2, cfg);

            % Update attention module and regression head.
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

            % Update encoder after the warm-up stage.
            if ep > cfg.warmupEpochs

                genc = grads.enc;

                % Fine-tune only the last several convolutional layers.
                genc = local_mask_enc_grads_keep_last_conv(currEnc, genc, cfg.tune_last_conv);

                if cfg.clip_enc > 0
                    genc = local_clip_grads_table(genc, cfg.clip_enc);
                end

                [currEnc, avgG_enc, avgSqG_enc] = adamupdate(currEnc, genc, ...
                    avgG_enc, avgSqG_enc, iter, cfg.lr_enc, cfg.beta1, cfg.beta2);
            end
        end
    end

    % Inference on the held-out pack.
    raw_te = X_img(:, :, 1, te);
    dlX_te = dlarray(reshape(raw_te', [1 cfg.M 1 cfg.nCell]), 'SSCB');

    dlZ = predict(currEnc, dlX_te);
    Z_te = stripdims(squeeze(dlZ));

    [ypred_norm, w_te] = model_predict_head_aligned(Z_te, ...
        attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2, cfg);

    % Save cell-level attention scores.
    w_num = double(gather(extractdata(w_te)));
    Cell_Attn_All(:, te) = w_num(:);
    MaxAttn_List(te) = max(w_num);

    % Transform prediction back to the original SOH scale.
    val_norm = gather(extractdata(ypred_norm));
    yhat(te) = val_norm * sgY + muY;

    fprintf('[Fold %d] True=%.3f | Pred=%.3f\n', te, y_all(te), yhat(te));
end

%% 5) Evaluate prediction performance
err = yhat - y_all;

RMSE = sqrt(mean(err.^2));
MAE  = mean(abs(err));
MAPE = mean(abs(err ./ y_all));

sst = sum((y_all - mean(y_all)).^2);
sse = sum(err.^2);
R2  = 1 - sse / sst;

fprintf('\n========== Final results: fine-tuning with L2-SP ==========\n');
fprintf('R2   = %.4f\n', R2);
fprintf('RMSE = %.4f\n', RMSE);
fprintf('MAE  = %.4f\n', MAE);
fprintf('MAPE = %.4f\n', MAPE);

%% 6) Export prediction results
fprintf('\nExporting fine-tuning L2-SP prediction results...\n');

% Pack-level prediction results with maximum attention score.
y_all_col    = double(y_all(:));
yhat_col     = double(yhat(:));
err_col      = double(err(:));
abs_err_col  = abs(err_col);
max_attn_col = double(MaxAttn_List(:));

Export_SOH = table( ...
    (1:Npack)', ...
    SN_List, ...
    y_all_col, ...
    yhat_col, ...
    err_col, ...
    abs_err_col, ...
    max_attn_col, ...
    'VariableNames', { ...
    'Pack_Index', ...
    'Pack_ID', ...
    'True_SOH', ...
    'Predicted_SOH', ...
    'Error', ...
    'Abs_Error', ...
    'Max_Attention_Score'});

% Cell-level attention-score matrix.
cell_col_names = arrayfun(@(x) sprintf('Pack_%d', x), ...
    1:Npack, 'UniformOutput', false);

Export_Cell_Attn = array2table(Cell_Attn_All, ...
    'VariableNames', cell_col_names);

Export_Cell_Attn = [ ...
    table((1:cfg.nCell)', 'VariableNames', {'Cell_Index'}), ...
    Export_Cell_Attn];

% Model-level performance metrics.
Export_Metrics = table( ...
    {'FineTune_L2SP_Attn'}, ...
    double(R2), ...
    double(RMSE), ...
    double(MAE), ...
    double(MAPE), ...
    'VariableNames', { ...
    'Model_Type', ...
    'R2', ...
    'RMSE', ...
    'MAE', ...
    'MAPE'});

% Write CSV files.
writetable(Export_SOH, cfg.pred_csv);
writetable(Export_Cell_Attn, cfg.attn_csv);
writetable(Export_Metrics, cfg.metrics_csv);

% Save complete MATLAB result file for downstream analysis.
save(cfg.result_mat, ...
    'cfg', ...
    'cfgT', ...
    'SN_List', ...
    'y_all', ...
    'yhat', ...
    'err', ...
    'RMSE', ...
    'MAE', ...
    'MAPE', ...
    'R2', ...
    'MaxAttn_List', ...
    'Cell_Attn_All', ...
    'Export_SOH', ...
    'Export_Cell_Attn', ...
    'Export_Metrics');

fprintf('[Export] Export completed.\n');
fprintf('  - Pack-level prediction results: %s\n', cfg.pred_csv);
fprintf('  - Cell-level attention matrix:   %s\n', cfg.attn_csv);
fprintf('  - Performance metrics:           %s\n', cfg.metrics_csv);
fprintf('  - MATLAB result file:            %s\n', cfg.result_mat);

fprintf('\n================ P2 Exp05 completed ================\n');

%% Local helper functions
function [loss, grads] = model_grads(X, y, net, net_ref, attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2, cfg)

    % Compute loss and gradients for L2-SP fine-tuning.
    [Z_raw, ~] = predict(net, X);
    Z = stripdims(squeeze(Z_raw));

    [ypred, ~] = model_predict_head_aligned(Z, ...
        attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2, cfg);

    % Mean squared error loss.
    mse_loss = mean((ypred - y).^2);

    % L2 regularization on head weight matrices.
    l2_head_term = sum(attW1(:).^2) + ...
                   sum(attW2(:).^2) + ...
                   sum(headW1(:).^2) + ...
                   sum(headW2(:).^2);

    % L2-SP regularization on encoder parameters.
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

    % Total loss.
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

function [ypred, w] = model_predict_head_aligned(Z, attW1, attb1, attW2, attb2, headW1, headb1, headW2, headb2, cfg)
% Forward pass of the soft-min attention module and MLP regression head.

    H = relu(attW1 * Z + attb1);

    % Soft-min attention with ReLU activation.
    score = -(attW2 * H + attb2) / cfg.attn_tau;

    % Softmax normalization.
    score = score - max(score, [], 2);
    w = exp(score) ./ sum(exp(score), 2);

    % Attention-based pooling.
    zbag = Z * w.';

    % MLP regression head.
    h1 = relu(headW1 * zbag + headb1);
    ypred = headW2 * h1 + headb2;
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

        if isempty(gi)
            continue;
        end

        gi_data = extractdata(gi);
        ss = ss + sum(gi_data(:).^2);
    end

    gn = sqrt(ss + 1e-12);

    if gn > clipnorm

        scale = clipnorm / gn;

        for i = 1:height(gtab)

            gi = gtab.Value{i};

            if isempty(gi)
                continue;
            end

            gtab.Value{i} = gi * scale;
        end
    end
end

function gcell = local_clip_grads_cell(gcell, clipnorm)

    ss = 0;

    for i = 1:numel(gcell)

        gi = gcell{i};

        if isempty(gi)
            continue;
        end

        gi_data = extractdata(gi);
        ss = ss + sum(gi_data(:).^2);
    end

    gn = sqrt(ss + 1e-12);

    if gn > clipnorm

        scale = clipnorm / gn;

        for i = 1:numel(gcell)

            if isempty(gcell{i})
                continue;
            end

            gcell{i} = gcell{i} * scale;
        end
    end
end