%% Speaker Recognition System using MFCC and Vector Quantization
clear; clc; close all;

%% Paths for accessing training and testing datasets
trainDataPath = 'C:\Startup MATLAB\EEC201-Project-main\Data\Train_our'; % Change path as needed
testDataPath = 'C:\Startup MATLAB\EEC201-Project-main\Data\Test_our';    % Change path as needed

% List audio files
trainAudioFiles = dir(fullfile(trainDataPath, '*.wav'));
testAudioFiles = dir(fullfile(testDataPath, '*.wav'));

% Choosing a specific number of speakers/samples
numTrainSpeakers = 11; % Number of training speakers
numTestSamples = 8;   % Number of testing samples
trainAudioFiles = trainAudioFiles(1:numTrainSpeakers); 
testAudioFiles = testAudioFiles(1:numTestSamples);    

% Defining feature extraction parameters
mfccCoeffCount = 26;  
clusterCount = 8;  
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
    frameSize = 256;  % N
    overlapSize = 70;  % M
    fftPoints = 256;

    [stftData, ~, ~] = stft(emphasizedSignal, sampleRate, 'Window', hamming(frameSize), ...
                            'OverlapLength', frameSize - overlapSize, 'FFTLength', fftPoints);
    powerSpectrum = (abs(stftData).^2) ./ fftPoints; 

    % Extracting MFCC features
    mfccFeatures = getMFCC(powerSpectrum, mfccCoeffCount, 40, 0, sampleRate, fftPoints);
    
    % Generating the codebook using vector quantization
    speakerCodebooks(speakerIdx, :, :) = applyVectorQuantization(mfccFeatures, clusterCount);
end
disp('Training phase completed.');

%% Processing test data and matching (Original Test Set)
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

    % Comparing features with each speaker's codebook
    for speakerIdx = 1:numTrainSpeakers
        distances = computeEuclideanDistance(mfccFeatures, squeeze(speakerCodebooks(speakerIdx, :, :)));
        similarityMatrix(testIdx, speakerIdx) = sum(min(distances, [], 2)); 
    end

    % Normalize similarity values for each test sample
    similarityMatrix(testIdx, :) = similarityMatrix(testIdx, :) ./ sum(similarityMatrix(testIdx, :));
end
disp('Testing phase completed.');

%% Displaying results for the original test set
disp('Matching Test Files to Closest Training Files:');
disp('Distance Matrix:');
disp(similarityMatrix)

correctMatches = 0;
for testIdx = 1:numTestSamples
    [~, closestSpeakerIdx] = min(similarityMatrix(testIdx, :));
    
    % Get the base file name (without extension) for both test and training files
    testName = getBaseName(testAudioFiles(testIdx).name);
    trainName = getBaseName(trainAudioFiles(closestSpeakerIdx).name);
    
    fprintf('Test file %s (Base: %s) is matching to Training file %s (Base: %s)\n', ...
        testAudioFiles(testIdx).name, testName, trainAudioFiles(closestSpeakerIdx).name, trainName);
    
    if strcmpi(testName, trainName)
        correctMatches = correctMatches + 1;
    end
end
fprintf('Accuracy on the original test set: %.2f%%\n', (correctMatches/numTestSamples)*100);

%% Notch Filter Robustness Testing
% In this section, the test signals are passed through notch filters at different center frequencies.
% This simulates scenarios where distinct features of the original voice signal may be suppressed.
notchFrequencies = [500, 1000, 1500, 2000];  % in Hz (adjust as needed)
numNotch = length(notchFrequencies);
accuracyNotch = zeros(numNotch, 1);

disp('Starting robustness testing with notch filters...');
for notchIdx = 1:numNotch
    curNotchFreq = notchFrequencies(notchIdx);
    correctMatchesNotch = 0;
    
    % Loop through each test sample
    for testIdx = 1:numTestSamples
        [testSignal, sampleRate] = audioread(fullfile(testDataPath, testAudioFiles(testIdx).name));
        
        % Preprocess the test signal
        testSignal = preprocessAudio(testSignal);
        testSignal = extractActiveSegment(testSignal, sampleRate);
        
        % Design the notch filter for current notch frequency
        % Normalize the notch frequency with respect to Nyquist (sampleRate/2)
        Wo = curNotchFreq / (sampleRate/2);
        Q = 50;  % Quality factor (adjustable)
        BW = Wo / Q;  
        [b, a] = iirnotch(Wo, BW);
        
        % Apply the notch filter to the test signal
        filteredSignal = filter(b, a, testSignal);
        
        % Continue processing: apply pre-emphasis
        emphasizedTestSignal = filter([1 -preEmphasisFactor], 1, filteredSignal);

        % STFT for filtered test signal
        [stftData, ~, ~] = stft(emphasizedTestSignal, sampleRate, 'Window', hamming(frameSize), ...
                                'OverlapLength', frameSize - overlapSize, 'FFTLength', fftPoints);
        powerSpectrum = (abs(stftData).^2) ./ fftPoints;
        
        % Extract MFCC features from the notch filtered signal
        mfccFeatures = getMFCC(powerSpectrum, mfccCoeffCount, 40, 0, sampleRate, fftPoints);
        
        % Compare with each speaker's codebook
        distanceVec = zeros(1, numTrainSpeakers);
        for speakerIdx = 1:numTrainSpeakers
            distances = computeEuclideanDistance(mfccFeatures, squeeze(speakerCodebooks(speakerIdx, :, :)));
            distanceVec(speakerIdx) = sum(min(distances, [], 2)); 
        end
        
        % Normalize the distances
        distanceVec = distanceVec / sum(distanceVec);
        [~, closestSpeakerIdx] = min(distanceVec);
        
        % Get the base names for matching comparison
        testName = getBaseName(testAudioFiles(testIdx).name);
        trainName = getBaseName(trainAudioFiles(closestSpeakerIdx).name);
        
        if strcmpi(testName, trainName)
            correctMatchesNotch = correctMatchesNotch + 1;
        end
    end
    
    % Calculate accuracy for the current notch frequency
    accuracyNotch(notchIdx) = (correctMatchesNotch / numTestSamples) * 100;
    fprintf('Notch Filter at %d Hz: Accuracy = %.2f%%\n', curNotchFreq, accuracyNotch(notchIdx));
end

%% Reporting Robustness
disp('Robustness Evaluation Summary:');
for notchIdx = 1:numNotch
    fprintf('Notch Frequency: %d Hz  ->  Accuracy: %.2f%%\n', notchFrequencies(notchIdx), accuracyNotch(notchIdx));
end

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
    frameDuration = 0.025; % 25 ms frames
    hopDuration = 0.010;   % 10 ms hop
    frameLength = round(frameDuration * fs);
    hopLength = round(hopDuration * fs);
    
    frames = buffer(audioSignal, frameLength, frameLength-hopLength, 'nodelay');
    energy = sum(frames.^2);
    
    threshold = 0.20 * max(energy);
    activeFrames = find(energy > threshold);
    
    if isempty(activeFrames)
        activeSignal = audioSignal;
    else
        startFrame = activeFrames(1);
        endFrame = activeFrames(end);
        startSample = (startFrame - 1) * hopLength + 1;
        endSample = min(length(audioSignal), (endFrame - 1) * hopLength + frameLength);
        activeSignal = audioSignal(startSample:endSample);
    end
end

function mfccData = getMFCC(powerSpec, coeffs, filters, startFreq, sampleRate, fftSize)
    powerSpec(powerSpec == 0) = eps;  
    melFilterBank = createMelFilter(filters, startFreq, sampleRate, fftSize);
    numFreqBins = floor(fftSize / 2) + 1;
    filteredSpec = melFilterBank * powerSpec(1:numFreqBins, :);
    maxCap = 1e10;  
    filteredSpec(filteredSpec > maxCap) = maxCap;
    filteredSpec(filteredSpec == 0) = eps;
    filteredSpec = 20 * log10(filteredSpec);
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

function baseName = getBaseName(filename)
    [~, baseName, ~] = fileparts(filename);
end
