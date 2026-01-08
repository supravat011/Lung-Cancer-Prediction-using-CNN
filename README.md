# Lung Cancer Prediction using CNN and Transfer Learning

A deep learning-based system for automated lung cancer classification from CT scan images using Convolutional Neural Networks (CNN) and Transfer Learning with the Xception architecture.

## 🎯 Overview

This project implements an AI-powered diagnostic tool that classifies lung CT scan images into four categories:
- **Normal** - No cancer detected
- **Adenocarcinoma** - A type of non-small cell lung cancer
- **Large Cell Carcinoma** - Aggressive lung cancer type  
- **Squamous Cell Carcinoma** - Cancer in the lining of airways

The model achieves **~93% accuracy** using transfer learning with the Xception architecture pre-trained on ImageNet.

## ✨ Features

- ✅ **High Accuracy**: 93% classification accuracy on test data
- ✅ **Transfer Learning**: Leverages Xception model pre-trained on ImageNet
- ✅ **Easy to Use**: Simple command-line interface for predictions
- ✅ **Fast Inference**: ~1-2 seconds per image
- ✅ **Confidence Scores**: Provides probability scores for all classes
- ✅ **Visual Output**: Displays predictions with matplotlib

## 🚀 Quick Start

### Prerequisites

- **Python 3.11 or 3.12** (TensorFlow doesn't support Python 3.14 yet)
- 4GB+ RAM recommended
- Internet connection (for first-time model download)

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   # If you have Python 3.11
   py -3.11 -m pip install -r requirements.txt
   
   # Or if Python 3.11 is your default
   pip install -r requirements.txt
   ```

3. **Run a prediction:**
   ```bash
   py -3.11 predict.py "dataset/test/normal/10.png"
   ```

### Usage Options

#### Option 1: Interactive Mode
```bash
py -3.11 predict.py
```
Then enter image paths when prompted.

#### Option 2: Command Line
```bash
# Single image
py -3.11 predict.py "path/to/image.png"

# Multiple images
py -3.11 predict.py "image1.png" "image2.png" "image3.png"
```

#### Option 3: One-Click (Windows)
Double-click `run_prediction.bat` and follow the prompts.

## 📊 Example Output

```
============================================================
Image: 10.png
Predicted Class: Normal
Confidence: 92.82%
============================================================

All Class Probabilities:
  Adenocarcinoma                :   4.23%
  Large Cell Carcinoma          :   2.48%
  Normal                        :  92.82%
  Squamous Cell Carcinoma       :   0.46%
```

## 🏗️ Project Structure

```
.
├── predict.py                    # Main inference script
├── requirements.txt              # Python dependencies
├── best_model.hdf5              # Pre-trained model weights (80MB)
├── dataset/                     # CT scan images
│   ├── train/                   # Training images
│   ├── valid/                   # Validation images
│   └── test/                    # Test images
├── Lung Cancer Pred.ipynb       # Training notebook
├── Lung Cancer Prediction.py    # Full training script
├── run_prediction.bat           # Windows launcher
├── example_usage.py             # Demo script
├── RUN_INSTRUCTIONS.md          # Detailed usage guide
└── README.md                    # This file
```

## 🧠 Model Architecture

- **Base Model**: Xception (pre-trained on ImageNet)
- **Custom Layers**: 
  - GlobalAveragePooling2D
  - Dense layer (4 units, softmax activation)
- **Input Size**: 350×350×3 RGB images
- **Output**: 4-class probability distribution
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## 📈 Training Details

The model was trained using:
- **Framework**: TensorFlow/Keras
- **Transfer Learning**: Xception with frozen weights
- **Data Augmentation**: Horizontal flips, rescaling
- **Callbacks**: Learning rate reduction, early stopping, model checkpointing
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 8
- **Training Accuracy**: ~93%
- **Validation Accuracy**: ~93%

## 📦 Dependencies

- TensorFlow >= 2.10.0
- Keras >= 2.10.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- Pillow >= 9.0.0

All dependencies are listed in `requirements.txt`.

## 📁 Dataset

The dataset consists of chest CT scan images organized into four classes. The images are split into:
- **Training set**: For model training
- **Validation set**: For hyperparameter tuning
- **Test set**: For final evaluation

Dataset source: [Chest CT Scan Images on Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

## 🔬 How It Works

1. **Image Preprocessing**: Input images are resized to 350×350 pixels and normalized
2. **Feature Extraction**: The Xception model extracts high-level features
3. **Classification**: A dense layer maps features to 4 cancer type probabilities
4. **Prediction**: The class with highest probability is selected

## 🎓 Use Cases

- **Medical Research**: Assist in lung cancer detection studies
- **Educational Tool**: Learn about deep learning in medical imaging
- **Diagnostic Support**: Preliminary screening tool (not for clinical diagnosis)
- **AI Development**: Base for building more advanced medical AI systems

## ⚠️ Important Notes

> **Disclaimer**: This is a research/educational project and should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

- The model was trained on specific CT scan datasets
- Performance may vary with different imaging equipment or protocols
- Best results with images similar to the training data
- Regular model updates and validation are recommended for production use

## 🛠️ Troubleshooting

### "TensorFlow cannot be installed"
- Make sure you're using Python 3.11 or 3.12
- Python 3.14 is not yet supported by TensorFlow

### "Model file not found"
- Ensure `best_model.hdf5` is in the same directory as `predict.py`

### "Image file not found"
- Use absolute paths or paths relative to the current directory
- Check that the image file exists and has the correct extension

For more help, see `RUN_INSTRUCTIONS.md`.

## 📝 Files Description

- **predict.py**: Standalone inference script with interactive and CLI modes
- **best_model.hdf5**: Trained model weights (Xception + custom layers)
- **Lung Cancer Pred.ipynb**: Jupyter notebook with full training pipeline
- **Lung Cancer Prediction.py**: Python script version of training code
- **requirements.txt**: All required Python packages
- **RUN_INSTRUCTIONS.md**: Comprehensive usage guide
- **run_prediction.bat**: Windows batch file for easy execution
- **example_usage.py**: Demo script showing batch prediction

## 🤝 Contributing

This is a personal project. If you have suggestions or improvements, feel free to fork and modify for your own use.

## 📧 Contact

For questions or collaboration opportunities, please reach out through the repository.

---

**Built with ❤️ using TensorFlow and Keras**

*Last Updated: January 2026*
