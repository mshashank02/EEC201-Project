%% Speaker Recognition System using MFCC and Vector Quantization
% The system learns from a set of training speakers and attempts to recognize 
% the speakers of test files by comparing extracted speech features.

clear; clc; close all;

%% Paths for accessing training and testing datasets
trainDataPath = 'E:\Speaker-Recognition-and-Preprocessing--main\Data\Train_our';
testDataPath = 'E:\Speaker-Recognition-and-Preprocessing--main\Data\Test_our';

trainAudioFiles = dir(fullfile(trainDataPath, '*.wav'));
testAudioFiles = dir(fullfile(testDataPath, '*.wav'));

% Choosing only a specific number of train and test speakers
numTrainSpeakers = 11;  
numTestSamples = 8;  
trainAudioFiles = trainAudioFiles(1:numTrainSpeakers); 
testAudioFiles = testAudioFiles(1:numTestSamples);    

% Defining feature extraction parameters
mfccCoeffCount = 19;  
clusterCount = 8;  
preEmphasisFactor = 0.99;  

% Allocating space for speaker codebooks
speakerCodebooks = zeros(numTrainSpeakers, mfccCoeffCount, clusterCount); 

%% Processing training data - extracting speaker features and building codebooks
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
disp('Training phase completed.');

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

function mfccData = getMFCC(powerSpec, coeffs, filters, startFreq, sampleRate, fftSize)
    % Creating Mel filterbank and applying it to the power spectrum
    melFilterBank = createMelFilter(filters, startFreq, sampleRate, fftSize);
    filteredSpec = melFilterBank * powerSpec(1:257, :);
    filteredSpec(filteredSpec == 0) = realmin;
    filteredSpec = 20 * log10(filteredSpec);

    % Applying discrete cosine transform (DCT) to obtain MFCCs
    mfccData = dct(filteredSpec, "Type", 2);
    mfccData = mfccData(2:coeffs+1, :);
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
    endMel = 1125 * log(1 + sampleRate / 2 / 700);
    melPoints = linspace(startMel, endMel, filters + 2);
    hzPoints = 700 * (exp(melPoints / 1125) - 1);
    
    binPositions = floor((fftSize + 1) * hzPoints / sampleRate);
    melFilters = zeros(filters, floor(fftSize / 2 + 1));

    for m = 2:(filters + 1)
        leftEdge = max(1, binPositions(m-1));
        rightEdge = min(size(melFilters, 2) - 1, binPositions(m+1));

        for k = leftEdge:rightEdge
            melFilters(m-1, k+1) = max(0, (k - binPositions(m-1)) / (binPositions(m) - binPositions(m-1)));
        end
    end
end
