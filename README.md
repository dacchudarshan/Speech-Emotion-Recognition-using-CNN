# Speech Emotion Recognition using CNN

A deep learning project that recognizes human emotions from speech audio using Convolutional Neural Networks (CNN).

## Project Overview

This project implements an emotion recognition system that analyzes audio files and classifies them into six emotional categories:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**

The model uses audio feature extraction and a 1D CNN architecture for classification.

## Dataset

The project combines audio data from multiple speech emotion datasets:
- **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- **RAVDESS** (Ryerson Audio-Visual Emotion Database and Speech Dataset)
- **SAVEE** (Surrey Audio-Visual Expressed Emotion)
- **TESS** (Toronto Emotional Speech Set)

### Current Data
- Total samples: **29,768** audio files
- Features extracted: **2,376 per audio file** (after augmentation and feature extraction)
- Data file: `processed_data.csv`

## Features Extracted

For each audio file, the following features are extracted:
- **Zero Crossing Rate (ZCR)** - Frequency of sign changes in the audio signal
- **RMSE (Root Mean Square Energy)** - Audio loudness measure
- **MFCC (Mel-Frequency Cepstral Coefficients)** - Speech characteristic features

### Data Augmentation
Each audio file is augmented with:
1. Original audio
2. Noise-added version
3. Pitch-shifted version
4. Pitch-shifted + Noise-added version

This provides 4x the training samples for better model generalization.

## Model Architecture

**CNN-Based Deep Learning Model**

```
- Conv1D Layer (512 filters, kernel=5)
  └─ BatchNormalization
  └─ MaxPooling1D (pool_size=5)

- Conv1D Layer (512 filters, kernel=5)
  └─ BatchNormalization
  └─ MaxPooling1D (pool_size=5)

- Conv1D Layer (256 filters, kernel=5)
  └─ BatchNormalization
  └─ MaxPooling1D (pool_size=5)

- Conv1D Layer (256 filters, kernel=3)
  └─ BatchNormalization
  └─ MaxPooling1D (pool_size=5)

- Conv1D Layer (128 filters, kernel=3)
  └─ BatchNormalization
  └─ MaxPooling1D (pool_size=3)

- Flatten Layer
- Dense Layer (512 units, ReLU)
  └─ BatchNormalization
- Dense Layer (6 units, Softmax)
```

**Model Statistics:**
- Total Parameters: **7,192,710**
- Trainable Parameters: **7,188,358**
- Model Size: **27.44 MB**

## Model Performance

- **Test Accuracy:** 17.2%
- **Test Loss:** 1.794
- **Best Performing:** Fear emotion (48% recall)
- **Most Challenging:** Disgust (1% recall), Neutral (2% recall)

### Classification Report
```
              precision    recall  f1-score   support
      angry       0.19      0.25      0.21      1057
    disgust       0.13      0.01      0.02      1042
       fear       0.18      0.48      0.26      1037
      happy       0.28      0.08      0.13       978
    neutral       0.07      0.02      0.03       842
        sad       0.14      0.15      0.14       998
```

## Requirements

### Dependencies
- Python 3.10+
- numpy
- pandas
- librosa (audio processing)
- matplotlib
- seaborn
- plotly
- scikit-learn
- tensorflow/keras
- IPython

### Installation

```bash
pip install numpy pandas librosa matplotlib seaborn plotly scikit-learn tensorflow
```

## Usage

### Running the Notebook

```bash
jupyter notebook app.ipynb
```

The notebook executes in sequence:

1. **Data Loading** - Loads audio files from multiple datasets
2. **Feature Extraction** - Extracts audio features with augmentation
3. **Data Preprocessing** - Cleans, scales, and splits the data
4. **Model Building** - Constructs the CNN architecture
5. **Model Training** - Trains with callbacks for early stopping and learning rate reduction
6. **Evaluation** - Generates confusion matrix and classification reports

### Training Parameters

```python
EPOCHS = 50
BATCH_SIZE = 64
```

**Callbacks:**
- Early Stopping: Monitors validation accuracy (patience=5)
- Learning Rate Reduction: Reduces LR on plateau (patience=3, factor=0.5)

## File Descriptions

| File | Description |
|------|-------------|
| `app.ipynb` | Main notebook with complete pipeline |
| `lll.ipynb` | Additional analysis/testing notebook |
| `processed_data.csv` | Pre-processed audio features and labels |
| `speech_emotion_cnn_model.h5` | Trained CNN model (HDF5 format) |
| `res_model.h5` | Alternative residual model checkpoint |
| `uploads/Crema/` | CREMA-D audio dataset directory |

## Model Improvements (Future Work)

1. **Data Balancing** - Address class imbalance (especially for disgust/neutral)
2. **Hyperparameter Tuning** - Optimize learning rate, batch size, and regularization
3. **Architecture Variants** - Experiment with ResNet, LSTM, or Attention mechanisms
4. **Additional Features** - Include spectral features, chromagram, etc.
5. **More Training Data** - Collect additional diverse audio samples
6. **Cross-validation** - Implement k-fold cross-validation
7. **Transfer Learning** - Fine-tune pre-trained models (e.g., VGGish, PANNs)

## Key Findings

- **Data Augmentation** significantly improves model robustness
- **Fear emotion** is the most recognizable (highest recall)
- **Class imbalance** affects performance on minority emotions
- **Batch normalization** helps training stability
- More diverse training data would improve generalization

## Notes

- The model was trained on a balanced augmented dataset
- Current low accuracy suggests potential improvements in feature engineering or model architecture
- Consider collecting more diverse audio samples with better emotion annotation quality
- Transfer learning approaches may provide better performance

## Author

Created for speech emotion recognition research and analysis.

## License

[Specify your license here]

---

**Last Updated:** January 23, 2026
