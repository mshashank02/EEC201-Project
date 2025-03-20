%% Speaker Recognition System using MFCC and Vector Quantization
clear; clc; close all;

%% Paths for accessing training and testing datasets
trainDataPath = 'C:\Startup MATLAB\EEC201-Project-main\Data\Train_our';
testDataPath = 'C:\Startup MATLAB\EEC201-Project-main\Data\Test_our';

trainAudioFiles = dir(fullfile(trainDataPath, '*.wav'));
testAudioFiles = dir(fullfile(testDataPath, '*.wav'));

numTrainSpeakers = 11;
numTestSamples = 8;
trainAudioFiles = trainAudioFiles(1:numTrainSpeakers);
testAudioFiles = testAudioFiles(1:numTestSamples);

mfccCoeffCount = 19;
clusterCount = 8;
preEmphasisFactor = 0.99;

speakerCodebooks = zeros(numTrainSpeakers, mfccCoeffCount, clusterCount);

%% Processing training data
disp('Extracting features from training speakers...');
for speakerIdx = 1:numTrainSpeakers
    [audioSignal, sampleRate] = audioread(fullfile(trainDataPath, trainAudioFiles(speakerIdx).name));
    
    % Applying pre-emphasis to highlight high-frequency components
    emphasizedSignal = filter([1 -preEmphasisFactor], 1, audioSignal);

    % Performing Short-Time Fourier Transform (STFT) for spectral analysis
    frameSize = 256;  
    overlapSize = 100;  
    fftPoints = 512;

    [stftData, ~, ~] = stft(emphasizedSignal, sampleRate, 'Window', hamming(frameSize), 'OverlapLength', frameSize - overlapSize, 'FFTLength', fftPoints);
    powerSpectrum = (abs(stftData).^2) ./ fftPoints; 

    % Extracting MFCC features from the power spectrum
    mfccFeatures = getMFCC(powerSpectrum, mfccCoeffCount, 40, 0, sampleRate, fftPoints);
    mfccFeatures = mfccFeatures - (mean(mfccFeatures) + 1e-8); 

    % Generating the codebook using vector quantization
    speakerCodebooks(speakerIdx, :, :) = applyVectorQuantization(mfccFeatures, clusterCount);
end



% Perform STFT and generate periodogram for different frame sizes
frameSizes = [128, 256, 512];
for fIdx = 1:length(frameSizes)
    N = frameSizes(fIdx);
    M = round(N / 3); % Frame increment

    figure;
    tiledlayout(3, ceil(numTrainSpeakers / 3)); % 3-row layout

    for speakerIdx = 1:numTrainSpeakers
        [audioSignal, sampleRate] = audioread(fullfile(trainDataPath, trainAudioFiles(speakerIdx).name));
        audioSignal = audioSignal(:, 1); % Ensure single channel

        % Compute STFT
        [s, f, t, ps] = spectrogram(audioSignal, hamming(N), M, N, sampleRate, 'yaxis');

        % Plot periodogram
        nexttile;
        imagesc(t * 1000, f, 10*log10(abs(ps)))
        axis xy;
        xlabel('Time (ms)');
        ylabel('Frequency (Hz)');
        title(['Speaker ', num2str(speakerIdx), ' - N=', num2str(N), ', M=', num2str(M)]);
        colorbar;
    end
    sgtitle(['STFT-Based Periodogram (N=', num2str(N), ', M=', num2str(M), ')']);
end

% Test 3: Natural Mel-spaced Filter Bank Response
p = 40;          % Number of mel filters (adjustable)
n = 512;         % FFT length (keeping it consistent)
fs = sampleRate; % Use actual sample rate from your audio files

% Define frequency axis
f = linspace(0, fs/2, 1+floor(n/2));

% Compute Mel filter bank (melfb should be defined as createMelFilter)
m = createMelFilter(p, 0, fs, n);

figure;
plot(f, full(m)');  % Convert sparse matrix to full for plotting
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Mel-Spaced Filter Bank Responses');
grid on;
ylim([0, 1]);  % Keep amplitude scaling consistent



% Compute and plot the spectrum before and after Mel filtering
% Ensure sufficient space in the layout
% Limit to first 5 speakers
numPlotSpeakers = min(5, numTrainSpeakers); 

figure;
tiledlayout(numPlotSpeakers, 2); % 5 rows, 2 columns (Before | After)

for speakerIdx = 1:numPlotSpeakers
    [audioSignal, sampleRate] = audioread(fullfile(trainDataPath, trainAudioFiles(speakerIdx).name));
    audioSignal = audioSignal(:, 1); % Ensure single channel
    
    % Compute STFT before Mel filtering
    [s, f, t, ps] = spectrogram(audioSignal, hamming(N), round(N/3), N, sampleRate, 'yaxis');
    powerSpec = abs(s).^2;
    
    % Plot spectrum before Mel filtering
    nexttile;
    imagesc(t, f, 10*log10(powerSpec));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title(['Speaker ', num2str(speakerIdx), ' - Before Mel Filtering']);
    colorbar;
    
    % Apply Mel filter bank
    melSpec = m * powerSpec(1:size(m, 2), :);
    
    % Plot spectrum after Mel filtering
    nexttile;
    imagesc(t, linspace(0, sampleRate / 2, size(melSpec, 1)), 10*log10(melSpec));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title(['Speaker ', num2str(speakerIdx), ' - After Mel Filtering']);
    colorbar;
end

sgtitle('Spectrum Before and After Mel Filtering (First 5 Speakers)');




% Initialize figures for plotting
% Plot Original Signals
figure;
tiledlayout(2, ceil(numTrainSpeakers / 2)); % 2 rows, ceil(numTrainSpeakers/2) columns
for speakerIdx = 1:numTrainSpeakers
    [audioSignal, sampleRate] = audioread(fullfile(trainDataPath, trainAudioFiles(speakerIdx).name));
audioSignal = audioSignal(:, 1); % Keep only the first channel

    timeVector = (0:length(audioSignal)-1) / sampleRate; 
    
    % Plot original waveform
    nexttile;
    plot(timeVector, audioSignal);
    xlabel('Time (seconds)');
    ylabel('Amplitude');
    title(['Original - ', trainAudioFiles(speakerIdx).name], 'Interpreter', 'none');
    grid on;
end
sgtitle('Original Signals of All Speakers'); % Super title for the figure




figure;
tiledlayout(2, ceil(numTrainSpeakers / 2)); % 2 rows, ceil(numTrainSpeakers/2) columns
for speakerIdx = 1:numTrainSpeakers
    [audioSignal, sampleRate] = audioread(fullfile(trainDataPath, trainAudioFiles(speakerIdx).name));
    
    % Apply pre-emphasis filter
    emphasizedSignal = filter([1 -preEmphasisFactor], 1, audioSignal);
    
    % Normalize
    emphasizedSignal = emphasizedSignal / max(abs(emphasizedSignal));
    timeVector = (0:length(emphasizedSignal)-1) / sampleRate;
    
    % Plot normalized waveform
    nexttile;
    plot(timeVector, emphasizedSignal);
    xlabel('Time (seconds)');
    ylabel('Amplitude');
    title(['Normalized - ', trainAudioFiles(speakerIdx).name], 'Interpreter', 'none');
    grid on;
end
sgtitle('Normalized Signals of All Speakers'); % Super title for the figure



% TEST 5: Visualizing MFCC Feature Space for Selected Speakers
disp('Performing TEST 5: Visualizing MFCC Feature Space using all extracted MFCC vectors');

% Select 3 specific speakers
selectedSpeakers = [3, 5, 10];  % Adjust indices if needed

% Choose two MFCC coefficients for visualization
mfccDim1 = 3; % MFCC Coefficient 3 (X-axis)
mfccDim2 = 4; % MFCC Coefficient 4 (Y-axis)

allMFCC = [];
speakerLabels = [];

figure;
hold on;
markers = {'x', 'o', 's'}; % Different markers for each speaker
colors = lines(length(selectedSpeakers)); % Generate distinct colors

for i = 1:length(selectedSpeakers)
    speakerIdx = selectedSpeakers(i);
    
    % Read the audio file
    [audioSignal, sampleRate] = audioread(fullfile(trainDataPath, trainAudioFiles(speakerIdx).name));
    audioSignal = audioSignal(:, 1); % Ensure single channel
    
    % Apply pre-emphasis
    emphasizedSignal = filter([1 -preEmphasisFactor], 1, audioSignal);

    % Perform STFT
    frameSize = 256;  
    overlapSize = 100;  
    fftPoints = 512;
    [stftData, ~, ~] = stft(emphasizedSignal, sampleRate, 'Window', hamming(frameSize), 'OverlapLength', frameSize - overlapSize, 'FFTLength', fftPoints);
    powerSpectrum = (abs(stftData).^2) ./ fftPoints;

    % Extract MFCC features
    mfccData = getMFCC(powerSpectrum, mfccCoeffCount, 40, 0, sampleRate, fftPoints);
    mfccData = mfccData'; % Transpose so rows = frames

    % Reduce number of vectors for better visualization
    numVectorsToPlot = min(500, size(mfccData, 1)); % Increase to show more points
    selectedIndices = randperm(size(mfccData, 1), numVectorsToPlot);
    mfccSubset = mfccData(selectedIndices, :);

    % Scatter plot with unique marker and color
    scatter(mfccSubset(:, mfccDim1), mfccSubset(:, mfccDim2), 30, colors(i, :), markers{i}, 'LineWidth', 1.2);

    % Store for combined legend
    allMFCC = [allMFCC; mfccSubset];
    speakerLabels = [speakerLabels; repmat(speakerIdx, numVectorsToPlot, 1)];
end

xlabel(['MFCC Coefficient ', num2str(mfccDim1)]);
ylabel(['MFCC Coefficient ', num2str(mfccDim2)]);
title('MFCC Feature Space of Selected Speakers');
legend(arrayfun(@(x) sprintf('Speaker %d', x), selectedSpeakers, 'UniformOutput', false));
grid on;
hold off;


% TEST 6: Visualizing VQ Codewords on MFCC Feature Space
disp('Performing TEST 6: Visualizing VQ Codewords on MFCC Feature Space');

% Create a new figure for TEST 6 (Separate from TEST 5)
figure;
hold on;
colors = lines(length(selectedSpeakers)); % Generate distinct colors
legendEntries = {}; % Store legend text

for i = 1:length(selectedSpeakers)
    speakerIdx = selectedSpeakers(i);

    % Extract all MFCC frames for this speaker
    mfccData = allMFCC(speakerLabels == speakerIdx, :);

    % Extract VQ Codewords from Codebook
    vqCodewords = squeeze(speakerCodebooks(speakerIdx, :, :))';

    % Plot MFCC Vectors (small markers)
    scatter(mfccData(:, mfccDim1), mfccData(:, mfccDim2), 10, colors(i, :), 'o', 'filled');

    % Plot VQ Codewords (larger square markers)
    scatter(vqCodewords(:, mfccDim1), vqCodewords(:, mfccDim2), 100, colors(i, :), 's', 'filled'); 

    % Add legend entry
    legendEntries{end+1} = sprintf('Speaker %d MFCC Vectors', speakerIdx);
    legendEntries{end+1} = sprintf('Speaker %d VQ Codewords', speakerIdx);
end

% Customize the plot
xlabel(['MFCC Coefficient ', num2str(mfccDim1)]);
ylabel(['MFCC Coefficient ', num2str(mfccDim2)]);
title('VQ Codewords and MFCC Vectors on MFCC Feature Space');
legend(legendEntries, 'Location', 'bestoutside');
grid on;
hold off;





%% Extracting features from test data and matching against trained speakers
disp('Analyzing test audio samples...');
similarityMatrix = zeros(numTestSamples, numTrainSpeakers);

for testIdx = 1:numTestSamples
    [testSignal, sampleRate] = audioread(fullfile(testDataPath, testAudioFiles(testIdx).name));

    % Applying pre-emphasis filter to maintain feature consistency
    emphasizedTestSignal = filter([1 -preEmphasisFactor], 1, testSignal);

    % Converting time-domain signal to frequency-domain representation
    [stftData, ~, ~] = stft(emphasizedTestSignal, sampleRate, 'Window', hamming(frameSize), 'OverlapLength', frameSize - overlapSize, 'FFTLength', fftPoints);
    powerSpectrum = (abs(stftData).^2) ./ fftPoints;

    % Extracting MFCC features for comparison
    mfccFeatures = getMFCC(powerSpectrum, mfccCoeffCount, 40, 0, sampleRate, fftPoints);
    mfccFeatures = mfccFeatures - (mean(mfccFeatures) + 1e-8);

    % Comparing the extracted features with each trained speaker's codebook
    for speakerIdx = 1:numTrainSpeakers
        distances = computeEuclideanDistance(mfccFeatures, squeeze(speakerCodebooks(speakerIdx, :, :)));
        similarityMatrix(testIdx, speakerIdx) = sum(min(distances, [], 2)); 
    end

    % Normalizing similarity values for better comparability
    similarityMatrix(testIdx, :) = similarityMatrix(testIdx, :) ./ sum(similarityMatrix(testIdx, :));
end
disp('Testing phase completed.');
disp('Distance Matrix:');
disp(similarityMatrix);

%% Displaying matched test and training file names
disp('Matching Test Files to Closest Training Files:');
for testIdx = 1:numTestSamples
    [~, closestSpeakerIdx] = min(similarityMatrix(testIdx, :));
    fprintf('Test file %s is matching to Training file %s\n', testAudioFiles(testIdx).name, trainAudioFiles(closestSpeakerIdx).name);
end







    %% FUNCTIONS
    function [mfccData, cepstrumData] = getMFCC(powerSpec, coeffs, filters, startFreq, sampleRate, fftSize)
        melFilterBank = createMelFilter(filters, startFreq, sampleRate, fftSize);
        filteredSpec = melFilterBank * powerSpec(1:257, :);
        filteredSpec(filteredSpec == 0) = realmin;
        filteredSpec = 20 * log10(filteredSpec);
        
        mfccData = dct(filteredSpec, 'Type', 2);
        mfccData = mfccData(2:coeffs+1, :);
        
        % Compute Cepstrum (TEST 4 Requirement)
        cepstrumData = idct(mfccData);
    end
    function quantizedCodebook = applyVectorQuantization(features, clusters)
        % Initializing cluster centers with feature mean
        epsilon = 0.01;  
        quantizedCodebook = mean(features, 2);
        distortionDelta = 1;
        numCentroids = 1;
    
        while numCentroids < clusters
            % Splitting clusters for better quantization
            newCodebook = repmat(quantizedCodebook, 1, 2);
            newCodebook(:, 1:2:end) = newCodebook(:, 1:2:end) * (1 + epsilon);
            newCodebook(:, 2:2:end) = newCodebook(:, 2:2:end) * (1 - epsilon);
            quantizedCodebook = newCodebook;
            numCentroids = size(quantizedCodebook, 2);
    
            % Refining cluster centers iteratively
            distMatrix = computeEuclideanDistance(features, quantizedCodebook);
            while abs(distortionDelta) > epsilon
                prevDistortion = mean(distMatrix);
                [~, nearestClusters] = min(distMatrix, [], 2);
                for i = 1:numCentroids
                    quantizedCodebook(:, i) = mean(features(:, nearestClusters == i), 2);
                end
                quantizedCodebook(isnan(quantizedCodebook)) = 0;
                distMatrix = computeEuclideanDistance(features, quantizedCodebook);
                distortionDelta = (prevDistortion - mean(distMatrix)) / prevDistortion;
            end
        end
    end
    
    function distMat = computeEuclideanDistance(matrix1, matrix2)
        % Computing Euclidean distance between two feature matrices
        [~, dim1] = size(matrix1);
        [~, dim2] = size(matrix2);
        distMat = zeros(dim1, dim2);
    
        for i = 1:dim1
            distMat(i, :) = sum((matrix2 - matrix1(:, i)).^2, 1);
        end
        distMat = sqrt(distMat);
    end
    
function melFilters = createMelFilter(filters, startFreq, sampleRate, fftSize)
    % Generating Mel filter banks based on the specified parameters
    startMel = 1125 * log(1 + startFreq / 700);
    endMel = 1125 * log(1 + (sampleRate / 2) / 700);
    melPoints = linspace(startMel, endMel, filters + 2);
    hzPoints = 700 * (exp(melPoints / 1125) - 1);
    
    binPositions = floor((fftSize + 1) * hzPoints / sampleRate);
    melFilters = zeros(filters, floor(fftSize / 2 + 1));

    for m = 1:filters
        leftEdge = binPositions(m);
        center = binPositions(m+1);
        rightEdge = binPositions(m+2);

        for k = leftEdge:center
            melFilters(m, k+1) = (k - leftEdge) / (center - leftEdge);
        end
        for k = center:rightEdge
            melFilters(m, k+1) = (rightEdge - k) / (rightEdge - center);
        end
    end
    
    % Normalize each filter
    melFilters = melFilters ./ max(melFilters, [], 2);
end
