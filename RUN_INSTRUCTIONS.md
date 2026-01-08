# 🚀 Quick Start Guide - Lung Cancer Prediction

This guide will help you run the lung cancer prediction model on your local machine.

## 📋 Prerequisites

- Python 3.8 or higher
- The pre-trained model file (`best_model.hdf5`) - already included in this project
- CT scan images to test (optional - you can use images from the dataset folder)

## 🔧 Installation

### Step 1: Install Required Libraries

Open PowerShell or Command Prompt in this directory and run:

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (deep learning framework)
- Keras (neural network API)
- NumPy (numerical computing)
- Matplotlib (visualization)
- Pillow (image processing)

**Note:** Installation may take a few minutes depending on your internet speed.

## 🎯 Running Predictions

You have **two ways** to run predictions:

### Method 1: Interactive Mode (Recommended for Beginners)

Simply run the script without any arguments:

```bash
python predict.py
```

The script will prompt you to enter image paths. You can:
- Enter the full path to an image file
- Type `quit` or `q` to exit

**Example:**
```
Enter image path (or 'quit' to exit): dataset/test/normal/image1.png
```

### Method 2: Command Line Mode (For Multiple Images)

Provide image paths as command line arguments:

```bash
python predict.py "path/to/image1.png" "path/to/image2.png"
```

**Example:**
```bash
python predict.py "dataset/test/normal/000119.png" "dataset/test/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000130.png"
```

## 📊 Understanding the Output

The script will display:

1. **Predicted Class**: The type of lung condition detected
   - Normal
   - Adenocarcinoma
   - Large Cell Carcinoma
   - Squamous Cell Carcinoma

2. **Confidence**: How confident the model is (0-100%)

3. **All Class Probabilities**: Probability scores for all classes

4. **Visual Display**: The image with the prediction overlaid

### Example Output:

```
============================================================
Image: test_image.png
Predicted Class: Normal
Confidence: 95.67%
============================================================

All Class Probabilities:
  Adenocarcinoma                : 2.15%
  Large Cell Carcinoma          : 1.23%
  Normal                        : 95.67%
  Squamous Cell Carcinoma       : 0.95%
```

## 🧪 Testing with Sample Images

If you have the dataset folder, you can test with existing images:

```bash
# Test with a normal image
python predict.py "dataset/test/normal/000119.png"

# Test with an adenocarcinoma image
python predict.py "dataset/test/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000130.png"

# Test multiple images at once
python predict.py "dataset/test/normal/000119.png" "dataset/test/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000130.png"
```

## 🎨 Supported Image Formats

The script supports common image formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff)

## ⚠️ Troubleshooting

### Issue: "Model file 'best_model.hdf5' not found!"
**Solution:** Make sure you're running the script from the project directory where `best_model.hdf5` is located.

### Issue: "Image file not found"
**Solution:** 
- Check the image path is correct
- Use absolute paths or paths relative to the current directory
- Ensure the image file exists

### Issue: TensorFlow installation errors
**Solution:** 
- Make sure you have Python 3.8 or higher
- Try installing with: `pip install tensorflow --upgrade`
- On Windows, you may need Visual C++ redistributables

### Issue: "No module named 'tensorflow'"
**Solution:** Run `pip install -r requirements.txt` again

## 📝 Notes

- The model expects CT scan images of size 350x350 pixels (it will automatically resize)
- The model was trained on chest CT scan images
- Best results are achieved with similar quality images
- The model achieves approximately **93% accuracy** on test data

## 🔬 Model Information

- **Architecture:** Xception (Transfer Learning)
- **Pre-trained on:** ImageNet
- **Fine-tuned on:** Lung Cancer CT Scan Dataset
- **Input Size:** 350x350x3 (RGB images)
- **Output Classes:** 4 (Normal, Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma)
- **Model Size:** ~80 MB

## 📚 Additional Resources

- Original Notebook: `Lung Cancer Pred.ipynb`
- Full Training Script: `Lung Cancer Prediction.py`
- Dataset: Available in the `dataset/` folder
- Kaggle Dataset: [Chest CT Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

## 🆘 Need Help?

If you encounter any issues:
1. Check that all dependencies are installed correctly
2. Verify the model file exists in the current directory
3. Ensure your image paths are correct
4. Check the image format is supported

---

**Happy Predicting! 🎉**
