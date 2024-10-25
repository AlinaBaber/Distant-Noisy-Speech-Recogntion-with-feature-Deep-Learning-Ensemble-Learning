# Distant-Noisy-Speech-Recogntion-feature-extraction-and-Classification-with-Deep-Learning-Ensemble-Learning

This project implements a Distant Speech Recognition (DSR) system using an ensemble of machine learning and deep learning models to classify speech under varying noise conditions. The system captures, preprocesses, extracts, and selects features from audio samples, then classifies speech data with models optimized for robust speech recognition.

## Project Overview

Distant Speech Recognition (DSR) seeks to enable accurate speech recognition in challenging environments with background noise and distant microphones, such as smart home devices and virtual assistants. This project builds on classic DSR techniques with a modern, robust feature extraction and classification pipeline.

## Data Sources

- **Without Noise and Background Noise Samples**:  
  - [TensorFlow Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
  - [Kaggle Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)
- **Average and High Noise Dataset Samples**:  
  - [Synthetic Speech Commands Dataset on Kaggle](https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset)

## Methodology
### Architecture
![diagram](https://github.com/user-attachments/assets/d621bdd4-8528-4544-9810-24fe8b87a617)

1. **Speech Signal Acquisition**:
   - Converts analog speech signals to digital with a sampling rate of 16 kHz.
   - Waveform divided into word chunks for processing.

2. **Preprocessing**:
   - Noise removal, frequency emphasis, and windowing using Hamming and low-pass filters.

3. **Feature Extraction**:
   - **Selected Features**:
      - **MFCC**: Captures short-term power spectrum for each sound.
      - **Mel-scaled Spectrogram**: Translates frequencies to Mel scale for human-perceived pitch.
      - **Polynomial (Poly) Feature**: Enables local polynomial approximations.
      - **Zero Crossing Rate (ZCR)**: Counts signal zero-crossing instances.
   - Other extracted features include LPCC, Rasta-PLP, Chroma variant CENS, and Tonnetz.

4. **Feature Selection & Transformation**:
   - Selects high-correlation features via correlation matrix analysis, reshaping data frames for model input.

5. **Classification**:
   - **Machine Learning Models**:
      - Random Forest
      - Support Vector Machine (SVM)
      - K-Nearest Neighbors (KNN)
   - **Deep Learning Models**:
      - Long Short-Term Memory (LSTM)
   - **Ensemble Voting Classifier**: Combines outputs from multiple models for improved classification.
## Results and Discussion

The classification models are evaluated on three noise levels: no noise, average noise, and high noise. Evaluation metrics include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Pearson Correlation Coefficient (R).

- **Best Performing Model**: LSTM with MFCC, Poly, Mel-scaled Spectrogram, and ZCR outperforms others in noisy conditions.
- **Notable Observations**:
   - SVM achieves high accuracy in low-noise conditions.
   - Voting Classifier balances model outputs for consistent results across noise levels.

### Prerequisites
- Python 3.x
- Libraries: `TensorFlow`, `Keras`, `scikit-learn`, `Librosa`

### License
This project is licensed under the MIT License - see the LICENSE file for details.
