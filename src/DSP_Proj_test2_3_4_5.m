%% Speaker Recognition System using MFCC and Vector Quantization
clear; clc; close all;

%% Paths for accessing training and testing datasets
trainDataPath = 'E:\Speaker-Recognition-and-Preprocessing--main\Data\Five Training';%Please change here the path of your training data set in your PC
testDataPath = 'E:\Speaker-Recognition-and-Preprocessing--main\Data\Five Test';%Please change here the path of your testing data set in your PC
%Please change here the number of speakers in training data set 
trainAudioFiles = dir(fullfile(trainDataPath, '*.wav'));
testAudioFiles = dir(fullfile(testDataPath, '*.wav'));

% Choosing only a specific number of train and test speakers
numTrainSpeakers = 23; %Please change here the number of speakers in training data set 
numTestSamples = 23;  %Please change here the number of speakers in testing data set 
trainAudioFiles = trainAudioFiles(1:numTrainSpeakers); 
testAudioFiles = testAudioFiles(1:numTestSamples);    

% Defining feature extraction parameters
mfccCoeffCount = 19;  
clusterCount = 8;  
preEmphasisFactor = 0.7;  

% Allocating space for speaker codebooks
speakerCodebooks = zeros(numTrainSpeakers, mfccCoeffCount, clusterCount); 

%% Processing training data - extracting speaker features and building codebooks
disp('Extracting features from training speakers...');
for speakerIdx = 1:numTrainSpeakers
    [audioSignal, sampleRate] = audioread(fullfile(trainDataPath, trainAudioFiles(speakerIdx).name));

    % Applying pre-emphasis
    emphasizedSignal = filter([1 -preEmphasisFactor], 1, audioSignal);

    % Performing Short-Time Fourier Transform (STFT)
    frameSize = 256;  
    overlapSize = 100;  
    fftPoints = 512;

    [stftData, ~, ~] = stft(emphasizedSignal, sampleRate, 'Window', hamming(frameSize), 'OverlapLength', frameSize - overlapSize, 'FFTLength', fftPoints);
    powerSpectrum = (abs(stftData).^2) ./ fftPoints; 

    % Extracting MFCC features
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

    % Applying pre-emphasis
    emphasizedTestSignal = filter([1 -preEmphasisFactor], 1, testSignal);

    % Converting time-domain signal to frequency-domain representation
    [stftData, ~, ~] = stft(emphasizedTestSignal, sampleRate, 'Window', hamming(frameSize), 'OverlapLength', frameSize - overlapSize, 'FFTLength', fftPoints);
    powerSpectrum = (abs(stftData).^2) ./ fftPoints;

    % Extracting MFCC features
    mfccFeatures = getMFCC(powerSpectrum, mfccCoeffCount, 40, 0, sampleRate, fftPoints);
    mfccFeatures = mfccFeatures - (mean(mfccFeatures) + 1e-8);

    % Comparing the extracted features with each trained speaker's codebook
    for speakerIdx = 1:numTrainSpeakers
        distances = computeEuclideanDistance(mfccFeatures, squeeze(speakerCodebooks(speakerIdx, :, :)));
        similarityMatrix(testIdx, speakerIdx) = sum(min(distances, [], 2)); 
    end

    % Normalizing similarity values
    similarityMatrix(testIdx, :) = similarityMatrix(testIdx, :) ./ sum(similarityMatrix(testIdx, :));
end
disp('Testing phase completed.');

%% Displaying matched test and training file names
disp('Matching Test Files to Closest Training Files:');
for testIdx = 1:numTestSamples
    [~, closestSpeakerIdx] = min(similarityMatrix(testIdx, :));
    fprintf('Test file %s is matching to Training file %s\n', testAudioFiles(testIdx).name, trainAudioFiles(closestSpeakerIdx).name);
end

%% FUNCTIONS

function mfccData = getMFCC(powerSpec, coeffs, filters, startFreq, sampleRate, fftSize)
    % Ensuring valid power spectrum
    powerSpec(powerSpec == 0) = eps;  

    % Generate Mel filterbank and apply it
    melFilterBank = createMelFilter(filters, startFreq, sampleRate, fftSize);
    filteredSpec = melFilterBank * powerSpec(1:257, :);

    % Cap extreme values to avoid Inf issues
    maxCap = 1e10;  
    filteredSpec(filteredSpec > maxCap) = maxCap;
    filteredSpec(filteredSpec == 0) = eps;

    % Convert to log scale
    filteredSpec = 20 * log10(filteredSpec);

    % Apply Discrete Cosine Transform (DCT)
    mfccData = dct(filteredSpec, "Type", 2);
    mfccData = mfccData(2:coeffs+1, :);
end

function quantizedCodebook = applyVectorQuantization(features, clusters)
    % Initialize quantized codebook
    numFeatures = size(features, 1);
    quantizedCodebook = zeros(numFeatures, clusters);

    % Compute initial centroid
    quantizedCodebook(:, 1) = mean(features, 2);

    % If only one centroid, duplicate it to match expected size
    if size(quantizedCodebook, 2) == 1
        quantizedCodebook = repmat(quantizedCodebook, 1, clusters);
    end
end

function melFilters = createMelFilter(filters, startFreq, sampleRate, fftSize)
    startMel = 1130 * log(1 + startFreq / 700);
    endMel = 1130 * log(1 + sampleRate / 2 / 700);
    melPoints = linspace(startMel, endMel, filters + 2);
    hzPoints = 700 * (exp(melPoints / 1130) - 1);
    
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

function distMat = computeEuclideanDistance(matrix1, matrix2)
    numVectors1 = size(matrix1, 2);
    numVectors2 = size(matrix2, 2);
    distMat = zeros(numVectors1, numVectors2);

    for i = 1:numVectors1
        distMat(i, :) = sqrt(sum((matrix2 - matrix1(:, i)).^2, 1));
    end
end
