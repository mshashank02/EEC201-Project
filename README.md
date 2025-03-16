# DSP Winter 2025 Project

## **Speaker Recognition using Digital Signal Processing**
Students: Shashank M, Sahishnu Raju Kakarlapudi

Team Name: Vocalists 

## **Instructions to Run the Code**

### **Test 1 - Initial Speaker Recognition on Given Speech Data**
**File:** `DSP_Speech_Winter_test1.m`

- This script contains code for **training and testing** on the initial speech files provided under the **GivenSpeech_Data** folder on Canvas.
- Specifically, it uses speech recordings where speakers say the word **"zero."**
- **Before running, update the data folder paths** in **lines 8 and 9**.
- Use the following dataset folders:
  - **Training Data:** `Train_our`
  - **Testing Data:** `Test_our`

### **Test 2, 3, 4, 5 - Extended Speaker Recognition**
**File:** `DSP_Proj_test2_3_4_5.m`

This script contains code for extended speaker recognition tests using additional datasets.
**Before running, update the data folder paths** in **lines 5 and 6**.
Adjust the **number of speakers** in **lines 12 and 13** based on the test being performed.

#### **Test Descriptions:**
- **Test 2:** Training and testing on previous year's **"Twelve"** speech files from the `2024StudentAudioRecording` folder.
- **Test 3:** Training and testing on previous year's **"Zero"** speech files from the `2024StudentAudioRecording` folder.
- **Test 4:** Training and testing on this year's (2025) students' **"Five"** speech files.
- **Test 5:** Training and testing on this year's (2025) students' **"Eleven"** speech files.


## **1. Introduction**
Speaker recognition is an essential application of digital signal processing (DSP) that enables the identification of individuals based on their speech characteristics. This project implements a speaker recognition system using Mel Frequency Cepstral Coefficients (MFCC) and vector quantization techniques. The goal is to extract and compare unique voice features for accurate speaker identification. By leveraging signal processing techniques, we aim to create a system capable of recognizing speakers based on their speech patterns, even with variations in volume, background noise, and pronunciation.

**Figure 1. Basic structures of speaker identification systems**

## **2. Problem Statement**
The primary objective of this project is to develop an efficient and robust speaker recognition system that can distinguish between different speakers based on their vocal characteristics. One of the key challenges in speaker recognition is ensuring that extracted speech features are distinct enough to enable accurate classification. Factors such as noise, pitch variations, and recording quality can influence recognition performance. 

In this preliminary phase, the system is tested on a dataset where speakers utter the word "zero." The results for this specific test case will be discussed in this report, while additional test cases will be included in the final submission.

## **3. Approach and Methodology**
The speaker recognition system follows a structured pipeline involving preprocessing, feature extraction, and classification.

### **3.1 Dataset and Preprocessing**
The dataset consists of recorded speech samples from multiple speakers, divided into training and test sets. The training set consists of speech recordings from 11 speakers, while the test set contains 8 speech samples.

To prepare the audio data for feature extraction, several preprocessing steps are applied:
- **Pre-Emphasis Filtering:** A high-pass filter is applied to the speech signal to amplify high-frequency components, improving the robustness of extracted features.
- **Framing and Windowing:** The speech signal is divided into overlapping frames to ensure a smooth spectral transition and maintain time-domain characteristics.
- **Short-Time Fourier Transform (STFT):** STFT is used to convert the time-domain signal into a frequency-domain representation, which provides a better understanding of the speech signal's spectral properties.

**Figure 2. Block diagram of the MFCC processor**

### **3.2 Feature Extraction using MFCC**
Feature extraction is a crucial step in speaker recognition, as it helps to isolate speaker-specific characteristics. The system extracts MFCCs from the speech signals because they closely mimic human auditory perception and are widely used in speech and speaker recognition tasks. 

The key steps in MFCC extraction include:
Computing the **power spectrum** of the speech signal.
Applying a **Mel filter bank** to focus on frequency ranges most relevant to human speech perception.
Applying a **logarithmic transformation** to enhance weaker speech features and normalize differences in intensity.
Using the **Discrete Cosine Transform (DCT)** to decorrelate features and concentrate energy into a smaller set of coefficients.

These MFCC features serve as the primary input for speaker recognition and classification.

**Figure 3. An example of mel-spaced filter bank**

### **3.3 Speaker Codebook Generation with Vector Quantization**
Once MFCC features are extracted, the system applies vector quantization to generate a speaker-specific codebook. This process involves:
**Clustering MFCC features** using an iterative approach to identify representative centroids for each speaker.
**Measuring similarity** between extracted features and stored codebooks using **Euclidean distance**.
**Assigning test samples** to the speaker whose codebook yields the lowest distance, indicating the closest match.

Vector qantization reduces the complexity of classification by representing each speaker's unique feature set with a limited number of representative points.

**Figure 4. Conceptual diagram illustrating vector quantization codebook formation**

## **4. Code Overview**
The speaker recognition system is implemented in MATLAB and follows a structured workflow:

First, the system loads audio files from the training and test datasets, applies pre-emphasis filtering, and extracts spectral features using STFT. The power spectrum of each speech signal is then used to compute MFCC features, which are normalized to ensure consistency across different recordings.

In the training phase, MFCC features are clustered using vector quntization, and a codebook is generated for each speaker. This codebook serves as a reference for matching test samples.

During testing, tthe system extracts MFCC features from the test speech samples and compares them against stored speaker codebooks using Euclidean distance. The speaker with the smallest distance is identified as the most probable match.

The final output of the system displays the predicted speaker for each test sample and computes the overall recognition accuracy.

## **5. Results and Analysis**
In this preliminary evaluation, the system was tested on a dataset where speakers pronounced the word "zero." The system achieved **100% recognition accuracy** for this specific test case. 

The results highlight the effectiveness of MFCC-based feature extraction and vector quantization for speaker recognition. The use of logarithmic scaling and DCT in the MFCC computation very much improves feature separability, leading to precise speaker classification. Additionally, by leveraging vector quantization, the system efficiently clusters and matches features, further enhancing accuracy.

In comparison to direct FFT-based recognition, the use of MFCC provides a more nicer representation of speech characteristics. FFT alone fails to capture speaker-specific nuances, whereas MFCC effectively emphasizes key speech features, resulting in superior classification performance.

While the system has demonstrated high accuracy in this test case, aditional evaluations are required to assess its robustness against variations in speech content, background noise, and speaker emotion. Future tests will have more words and diverse speaker conditions to validate its effectiveness further.

## **6. Conclusion and Future Work**
This preliminary implementation successfully demonstrates a speaker recognition system using MFCC and vector quantization, achieving 100% accuracy on the test case. The results confirm that MFCC-based feature extraction is highly effective for distinguishing speakers, while vector quantization provides a compact and efficient representation of speech features.

Moving forward, additional test cases will be included to evaluate the system's performance under varied conditions. Future enhancements may involve:
Expanding the dataset to include different spoken words and sentences.
Evaluating the system’s performance in noisy environments.
Exploring deep learning-based speaker embeddings for improved accuracy and scalability.
Implementing real-time speaker recognition applications for practical deployment.

**Figure 5. Flow diagram of LBG algorithm (Adopted from Rabiner and Juang, 1993)**
