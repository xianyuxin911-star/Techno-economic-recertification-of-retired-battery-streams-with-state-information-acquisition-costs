%% P1_Exp02_SourceDomain_CNNTraining_CV.m
% Purpose: Trains the source-domain CNN model for cell-level SOH prediction, evaluates 
% out-of-fold prediction performance using 5-fold cross-validation, saves the trained 
% source-domain model for downstream transfer learning, and generates Extended Data Fig. 1b.

clc; clear; close all;

%%  1. Configuration & Initialization

cfg = struct('kfold', 5, 'seed', 42, 'maxEpochs', 800, 'miniBatch', 16, ...
             'learnRate', 1e-3, 'weightDecay', 1e-4, 'dropout', 0.4, ...
             'valFrac', 0.15, 'valPatience', 50, 'plotTrain', false, ...
             'save_fig', true, 'save_png', true);

rng(cfg.seed);

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end

part_dir = script_dir;

output_data_dir    = fullfile(part_dir, 'Output', 'Data');
output_model_dir   = fullfile(part_dir, 'Output', 'Models');
output_results_dir = fullfile(part_dir, 'Output', 'Results');
figure_exd_dir     = fullfile(part_dir, 'Figures', 'Extended');

arrayfun(@(x) (exist(x{1}, 'dir') || mkdir(x{1})), ...
    {output_data_dir, output_model_dir, output_results_dir, figure_exd_dir});

dataset_file = fullfile(output_data_dir, 'P1_SourceDomain_Dataset.mat');
model_file = fullfile(output_model_dir, 'P1_SourceDomain_CNN_Model.mat');
pred_file = fullfile(output_results_dir, 'P1_SourceDomain_CVPredictions.csv');
metrics_file = fullfile(output_results_dir, 'P1_SourceDomain_CVMetrics.csv');

exd_fig_file = fullfile(figure_exd_dir, 'ExD01a_SourceDomainPredictionPerformance.fig');
exd_png_file = fullfile(figure_exd_dir, 'ExD01a_SourceDomainPredictionPerformance.png');

%%  2. Load Dataset

if ~exist(dataset_file, 'file')
    error('Source-domain dataset not found. Please run the source-domain Q-V preprocessing script first.');
end
S = load(dataset_file);
X = double(S.Train_X); 
Y = double(S.Train_Y(:));
[N, cfg.M] = size(X);

fprintf('[INFO] Dataset: N=%d, M=%d | SOH: [%.4f, %.4f]\n', N, cfg.M, min(Y), max(Y));
fprintf('[BASELINE] Mean predictor RMSE=%.5f\n', sqrt(mean((Y - mean(Y)).^2)));

%%  3. K-Fold Cross-Validation

cvp = cvpartition(N, 'KFold', cfg.kfold);
yhat_all = nan(N, 1);
metrics_mat = zeros(cfg.kfold, 5); % [Fold, RMSE, MAE, R2, nTest]

for f = 1:cfg.kfold
    idxTr = training(cvp, f); idxTe = test(cvp, f);
    
    % Global standardization based on training fold
    muX = mean(X(idxTr,:), 'all'); sgX = max(std(X(idxTr,:), 0, 'all'), 1e-12);
    muY = mean(Y(idxTr));          sgY = max(std(Y(idxTr)), 1e-12);
    
    XtrN = (X(idxTr,:) - muX) / sgX;
    Xte4 = reshape(((X(idxTe,:) - muX) / sgX)', [1, cfg.M, 1, sum(idxTe)]);
    YtrN = (Y(idxTr) - muY) / sgY;

    % Validation split
    cvp2 = cvpartition(sum(idxTr), 'Holdout', cfg.valFrac);
    idTr = training(cvp2, 1); idVa = test(cvp2, 1);
    
    Xtrain4 = reshape(XtrN(idTr,:)', [1, cfg.M, 1, sum(idTr)]);
    Xval4   = reshape(XtrN(idVa,:)', [1, cfg.M, 1, sum(idVa)]);

    % Train fold model
    opts = get_train_opts(Xval4, YtrN(idVa), cfg);
    net  = trainNetwork(Xtrain4, YtrN(idTr), build_cnn2d_reg(cfg), opts);

    % Predict and scale back
    Yhat = double(predict(net, Xte4, 'MiniBatchSize', cfg.miniBatch)) * sgY + muY;
    yhat_all(idxTe) = Yhat;

    [rmse, mae, r2] = calc_metrics(Y(idxTe), Yhat);
    metrics_mat(f, :) = [f, rmse, mae, r2, sum(idxTe)];
    fprintf('[Fold %d] RMSE=%.5f | R2=%.4f\n', f, rmse, r2);
end

% Summarize and Convert to Table
foldMetrics = array2table(metrics_mat, 'VariableNames', {'Fold','RMSE','MAE','R2','nTest'});
fprintf('\n========== CV Summary ==========\nRMSE = %.5f ± %.5f | R2 = %.4f\n', ...
    mean(metrics_mat(:,2)), std(metrics_mat(:,2)), mean(metrics_mat(:,4), 'omitnan'));

%%  4. Train Final Model & Export Results

fprintf('\nTraining the final source-domain model...\n');
muX_full = mean(X, 'all'); sgX_full = max(std(X, 0, 'all'), 1e-12);
muY_full = mean(Y);        sgY_full = max(std(Y), 1e-12);

X4F = reshape(((X - muX_full) / sgX_full)', [1, cfg.M, 1, N]);
YnF = (Y - muY_full) / sgY_full;

cvpF = cvpartition(N, 'Holdout', cfg.valFrac);
optsFull = get_train_opts(X4F(:,:,:,test(cvpF,1)), YnF(test(cvpF,1)), cfg);
netFull  = trainNetwork(X4F(:,:,:,training(cvpF,1)), YnF(training(cvpF,1)), build_cnn2d_reg(cfg), optsFull);

save(model_file, 'netFull', 'cfg', 'muX_full', 'sgX_full', 'muY_full', 'sgY_full', 'foldMetrics', 'Y', 'yhat_all');
ExportResults = table((1:N)', Y, yhat_all, yhat_all - Y, 'VariableNames', {'Sample_Index', 'True_SOH', 'CV_Predicted_SOH', 'CV_Error'});
writetable(ExportResults, pred_file);
writetable(foldMetrics, metrics_file);
fprintf('[DONE] Model and results exported successfully.\n');

%% 5. Visualize Source-Domain Predictions (Restored Layout)

true_soh = ExportResults.True_SOH;
pred_soh = ExportResults.CV_Predicted_SOH;
error_val = ExportResults.CV_Error;

figure('Position', [150, 150, 900, 650], 'Color', 'w', 'Name', 'Source-domain SOH Prediction Results');

% Restored custom positioning for marginal histograms
pos_main  = [0.12, 0.12, 0.55, 0.55];
pos_top   = [0.12, 0.70, 0.55, 0.20];
pos_right = [0.75, 0.12, 0.18, 0.55];
pos_cbar  = [0.75, 0.78, 0.18, 0.03];

min_val = min([true_soh; pred_soh]) - 0.01;
max_val = max([true_soh; pred_soh]) + 0.01;
max_err = max(abs(error_val));
if max_err < 1e-12, max_err = 1e-3; end

%% Main parity plot
ax_main = axes('Position', pos_main);
hold on; grid on; box on;
plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5);
scatter(true_soh, pred_soh, 45, true_soh, 'filled', 'MarkerEdgeColor', [0.4 0.4 0.4], 'LineWidth', 0.5, 'MarkerFaceAlpha', 0.8);
xlabel('True SOH', 'FontSize', 12, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
ylabel('Predicted SOH', 'FontSize', 12, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
xlim([min_val, max_val]); ylim([min_val, max_val]);
set(gca, 'FontSize', 11, 'FontName', 'Times New Roman');
colormap(ax_main, turbo); clim(ax_main, [min(true_soh), max(true_soh)]);

%% SOH colorbar
c = colorbar(ax_main, 'Position', pos_cbar, 'Orientation', 'horizontal');
c.Label.String = 'SOH Value';
c.Label.FontName = 'Times New Roman';
c.Label.FontSize = 11;
c.Label.FontWeight = 'bold';
c.XAxisLocation = 'top';

%% True SOH distribution
ax_top = axes('Position', pos_top);
hold on; box on;
histogram(true_soh, 25, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'w', 'FaceAlpha', 0.7);
[f_true, xi_true] = ksdensity(true_soh);
plot(xi_true, f_true, 'Color', [0 0.4 0.7], 'LineWidth', 2);
xlim([min_val, max_val]); ax_top.XTickLabel = [];
ylabel('Density', 'FontSize', 11, 'FontName', 'Times New Roman');
title('Distribution of True SOH', 'FontSize', 13, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
set(gca, 'FontSize', 10, 'FontName', 'Times New Roman');

%% Prediction-error distribution
ax_right = axes('Position', pos_right);
hold on; grid on; box on;
histogram(error_val, 25, 'Normalization', 'pdf', 'Orientation', 'horizontal', 'FaceColor', [0.8 0.4 0.4], 'EdgeColor', 'w', 'FaceAlpha', 0.7);
[f_err, xi_err] = ksdensity(error_val);
plot(f_err, xi_err, 'Color', [0.7 0 0], 'LineWidth', 2);
yline(0, 'k--', 'LineWidth', 1.5);
ylim([-max_err, max_err]);
ylabel('Error (Pred - True)', 'FontSize', 11, 'FontName', 'Times New Roman');
xlabel('Density', 'FontSize', 11, 'FontName', 'Times New Roman');
title('Error Dist.', 'FontSize', 13, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
set(gca, 'FontSize', 10, 'FontName', 'Times New Roman');

drawnow;

if cfg.save_fig
    savefig(exd_fig_file);
    fprintf('Saved Extended Data figure: %s\n', exd_fig_file);
end

if cfg.save_png
    exportgraphics(gcf, exd_png_file, 'Resolution', 600);
    fprintf('Saved Extended Data PNG: %s\n', exd_png_file);
end

fprintf('Source-domain prediction-performance figure completed.\n');

%%  Local Functions

function opts = get_train_opts(Xval, Yval, cfg)
    opts = trainingOptions('adam', 'InitialLearnRate', cfg.learnRate, ...
        'MaxEpochs', cfg.maxEpochs, 'MiniBatchSize', cfg.miniBatch, ...
        'Shuffle', 'every-epoch', 'L2Regularization', cfg.weightDecay, ...
        'ValidationData', {Xval, Yval}, 'Verbose', false);
    if cfg.plotTrain, opts.Plots = 'training-progress'; end
    try opts.ValidationPatience = cfg.valPatience; catch; end
end

function layers = build_cnn2d_reg(cfg)
    layers = [
        imageInputLayer([1 cfg.M 1], 'Normalization', 'none', 'Name', 'in')
        convolution2dLayer([1 5], 16, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        leakyReluLayer(0.1, 'Name', 'lrelu1')
        maxPooling2dLayer([1 2], 'Stride', [1 2], 'Name', 'pool1')
        convolution2dLayer([1 5], 32, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        leakyReluLayer(0.1, 'Name', 'lrelu2')
        maxPooling2dLayer([1 2], 'Stride', [1 2], 'Name', 'pool2')
        convolution2dLayer([1 3], 64, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        leakyReluLayer(0.1, 'Name', 'lrelu3')
        globalAveragePooling2dLayer('Name', 'gap')
        fullyConnectedLayer(64, 'Name', 'fc1')
        leakyReluLayer(0.1, 'Name', 'lrelu4')
        dropoutLayer(cfg.dropout, 'Name', 'drop')
        fullyConnectedLayer(1, 'Name', 'fc_out')
        regressionLayer('Name', 'reg')
    ];
end

function [rmse, mae, r2] = calc_metrics(y, yhat)
    % Restored to safe if-else structure to prevent NaN * 0 issues
    y = double(y(:));
    yhat = double(yhat(:));
    err = yhat - y;
    
    rmse = sqrt(mean(err.^2));
    mae  = mean(abs(err));
    
    sst = sum((y - mean(y)).^2);
    sse = sum(err.^2);
    
    if sst < 1e-12
        r2 = NaN;
    else
        r2 = 1 - sse / sst;
    end
end
