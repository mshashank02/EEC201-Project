%Script for best implementation of Five and Twelve Single Dataset Speaker
%Recognition
%% Speaker Recognition System using MFCC and Vector Quantization
clear; clc; close all;

%% Paths for accessing training and testing datasets
trainDataPath = 'C:\Startup MATLAB\EEC201-Project-main\Data\Twelve-Training'; % Change path as needed
testDataPath = 'C:\Startup MATLAB\EEC201-Project-main\Data\Twelve-Testing';    % Change path as needed

% List audio files
trainAudioFiles = dir(fullfile(trainDataPath, '*.wav'));
testAudioFiles = dir(fullfile(testDataPath, '*.wav'));

% Choosing a specific number of speakers/samples
numTrainSpeakers = 18; % Number of training speakers
numTestSamples = 18;   % Number of testing samples
trainAudioFiles = trainAudioFiles(1:numTrainSpeakers); 
testAudioFiles = testAudioFiles(1:numTestSamples);    

% Defining feature extraction parameters
mfccCoeffCount = 26;  
clusterCount = 4;  
preEmphasisFactor = 0.99;  

% Allocating space for speaker codebooks
speakerCodebooks = zeros(numTrainSpeakers, mfccCoeffCount, clusterCount); 

%% Processing training data - extracting speaker features and building codebooks
disp('Extracting features from training speakers...');
for speakerIdx = 1:numTrainSpeakers
    [audioSignal, sampleRate] = audioread(fullfile(trainDataPath, trainAudioFiles(speakerIdx).name));
    
    % Data Preprocessing: Normalize and reduce noise
    audioSignal = preprocessAudio(audioSignal);
    
    % Extract only the active segment based on energy (this helps for digits like zero/twelve)
    audioSignal = extractActiveSegment(audioSignal, sampleRate);
    
    % Applying pre-emphasis
    emphasizedSignal = filter([1 -preEmphasisFactor], 1, audioSignal);

    % Performing Short-Time Fourier Transform (STFT)
    frameSize = 512;  % N
    overlapSize = 170;  % M
    fftPoints = 512;

    [stftData, ~, ~] = stft(emphasizedSignal, sampleRate, 'Window', hamming(frameSize), ...
                            'OverlapLength', frameSize - overlapSize, 'FFTLength', fftPoints);
    powerSpectrum = (abs(stftData).^2) ./ fftPoints; 

    % Extracting MFCC features
    mfccFeatures = getMFCC(powerSpectrum, mfccCoeffCount, 40, 0, sampleRate, fftPoints);
    % (Optionally remove mean if using CMVN later; avoid double normalization)
    % mfccFeatures = mfccFeatures - (mean(mfccFeatures) + 1e-8);
    
    % You might also try Cepstral Mean and Variance Normalization (CMVN) here—
    % if applied properly it can help, but be cautious if it over-normalizes.
    % mfccFeatures = cmvn(mfccFeatures);

    % Generating the codebook using vector quantization
    speakerCodebooks(speakerIdx, :, :) = applyVectorQuantization(mfccFeatures, clusterCount);
end
disp('Training phase completed.');

%% Processing test data and matching
disp('Analyzing test audio samples...');
similarityMatrix = zeros(numTestSamples, numTrainSpeakers);

for testIdx = 1:numTestSamples
    [testSignal, sampleRate] = audioread(fullfile(testDataPath, testAudioFiles(testIdx).name));
    
    % Data Preprocessing: Normalize and reduce noise
    testSignal = preprocessAudio(testSignal);
    
    % Extract only the active segment
    testSignal = extractActiveSegment(testSignal, sampleRate);
    
    % Applying pre-emphasis
    emphasizedTestSignal = filter([1 -preEmphasisFactor], 1, testSignal);

    % STFT for test signal
    [stftData, ~, ~] = stft(emphasizedTestSignal, sampleRate, 'Window', hamming(frameSize), ...
                            'OverlapLength', frameSize - overlapSize, 'FFTLength', fftPoints);
    powerSpectrum = (abs(stftData).^2) ./ fftPoints;

    % Extracting MFCC features
    mfccFeatures = getMFCC(powerSpectrum, mfccCoeffCount, 40, 0, sampleRate, fftPoints);
    % mfccFeatures = mfccFeatures - (mean(mfccFeatures) + 1e-8);
    % mfccFeatures = cmvn(mfccFeatures);

    % Comparing features with each speaker's codebook
    for speakerIdx = 1:numTrainSpeakers
        distances = computeEuclideanDistance(mfccFeatures, squeeze(speakerCodebooks(speakerIdx, :, :)));
        similarityMatrix(testIdx, speakerIdx) = sum(min(distances, [], 2)); 
    end

    % Normalize similarity values for each test sample
    similarityMatrix(testIdx, :) = similarityMatrix(testIdx, :) ./ sum(similarityMatrix(testIdx, :));
end
disp('Testing phase completed.');

%% Displaying results
disp('Matching Test Files to Closest Training Files:');
disp('Distance Matrix:');
disp(similarityMatrix)

correctMatches = 0;
for testIdx = 1:numTestSamples
    [~, closestSpeakerIdx] = min(similarityMatrix(testIdx, :));
    fprintf('Test file %s is matching to Training file %s\n', testAudioFiles(testIdx).name, trainAudioFiles(closestSpeakerIdx).name);
    if closestSpeakerIdx == testIdx
        correctMatches = correctMatches + 1;
    end
end

accuracy = (correctMatches / numTestSamples) * 100;
fprintf('Overall Accuracy: %.2f%%\n', accuracy);

%% FUNCTIONS

function processedSignal = preprocessAudio(audioSignal)
    % Remove DC offset
    audioSignal = audioSignal - mean(audioSignal);
    % Normalize amplitude to [-1, 1]
    maxVal = max(abs(audioSignal));
    if maxVal > 0
        audioSignal = audioSignal / maxVal;
    end
    processedSignal = audioSignal;
end

function activeSignal = extractActiveSegment(audioSignal, fs)
    % Use an energy-based endpoint detection to extract the high-energy region.
    % Parameters for energy calculation:
    frameDuration = 0.025; % 25 ms frames
    hopDuration = 0.010;   % 10 ms hop
    frameLength = round(frameDuration * fs);
    hopLength = round(hopDuration * fs);
    
    % Compute short-term energy per frame:
    frames = buffer(audioSignal, frameLength, frameLength-hopLength, 'nodelay');
    energy = sum(frames.^2);
    
    % Use a threshold relative to maximum energy (for example, 20% of max)
    threshold = 0.2 * max(energy);
    activeFrames = find(energy > threshold);
    
    if isempty(activeFrames)
        activeSignal = audioSignal;  % fallback if no active frames found
    else
        % Use the first and last active frame to define the region:
        startFrame = activeFrames(1);
        endFrame = activeFrames(end);
        startSample = (startFrame - 1) * hopLength + 1;
        endSample = min(length(audioSignal), (endFrame - 1) * hopLength + frameLength);
        activeSignal = audioSignal(startSample:endSample);
    end
end

function mfccData = getMFCC(powerSpec, coeffs, filters, startFreq, sampleRate, fftSize)
    % Ensure no zeros in the power spectrum
    powerSpec(powerSpec == 0) = eps;  
    % Generate Mel filterbank and apply it
    melFilterBank = createMelFilter(filters, startFreq, sampleRate, fftSize);
    numFreqBins = floor(fftSize / 2) + 1;
    filteredSpec = melFilterBank * powerSpec(1:numFreqBins, :);
    % Cap extreme values
    maxCap = 1e10;  
    filteredSpec(filteredSpec > maxCap) = maxCap;
    filteredSpec(filteredSpec == 0) = eps;
    % Log conversion
    filteredSpec = 20 * log10(filteredSpec);
    % Compute DCT to get MFCCs
    mfccData = dct(filteredSpec, "Type", 2);
    mfccData = mfccData(2:coeffs+1, :);
end

function quantizedCodebook = applyVectorQuantization(features, clusters)
    epsilon = 0.01;  
    quantizedCodebook = mean(features, 2);
    distortionDelta = 1;
    numCentroids = 1;
    
    while numCentroids < clusters
        newCodebook = repmat(quantizedCodebook, 1, 2);
        newCodebook(:, 1:2:end) = newCodebook(:, 1:2:end) * (1 + epsilon);
        newCodebook(:, 2:2:end) = newCodebook(:, 2:2:end) * (1 - epsilon);
        quantizedCodebook = newCodebook;
        numCentroids = size(quantizedCodebook, 2);
        
        distMatrix = computeEuclideanDistance(features, quantizedCodebook);
        while abs(distortionDelta) > epsilon
            prevDistortion = mean(distMatrix(:));
            [~, nearestClusters] = min(distMatrix, [], 2);
            for i = 1:numCentroids
                if any(nearestClusters == i)
                    quantizedCodebook(:, i) = mean(features(:, nearestClusters == i), 2);
                end
            end
            quantizedCodebook(isnan(quantizedCodebook)) = 0;
            distMatrix = computeEuclideanDistance(features, quantizedCodebook);
            distortionDelta = (prevDistortion - mean(distMatrix(:))) / prevDistortion;
        end
    end
end

function melFilters = createMelFilter(filters, startFreq, sampleRate, fftSize)
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

function distMat = computeEuclideanDistance(matrix1, matrix2)
    numVectors1 = size(matrix1, 2);
    numVectors2 = size(matrix2, 2);
    distMat = zeros(numVectors1, numVectors2);
    for i = 1:numVectors1
        distMat(i, :) = sqrt(sum((matrix2 - matrix1(:, i)).^2, 1));
    end
end
