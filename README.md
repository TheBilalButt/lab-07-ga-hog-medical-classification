# Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheBilalButt/lab-07-ga-hog-medical-classification/blob/main/Lab07_GA_HOG_Medical_Classification.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/TheBilalButt/lab-07-ga-hog-medical-classification)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PneumoniaMNIST-2E7D32)](https://github.com/TheBilalButt/lab-07-ga-hog-medical-classification)

**Student Name:** Muhammad Bilal Butt  
**Student ID:** FA23-BAI-042  
**Course:** Computer Vision  
**Assigned Dataset:** PneumoniaMNIST  
**Interactive Colab Notebook:** [Lab07_GA_HOG_Medical_Classification.ipynb](https://colab.research.google.com/github/TheBilalButt/lab-07-ga-hog-medical-classification/blob/main/Lab07_GA_HOG_Medical_Classification.ipynb)  

---

## 1. System Requirements & Data Specifications (SRDS)

To allow anyone viewing this project to reproduce the experiments quickly, the technical specifications are summarized below:

### Hardware and Software Environment
- Operating System: Windows 10/11 or Ubuntu 22.04 LTS (also fully compatible with Google Colab Linux runtime)
- Python Runtime: Python 3.10, 3.11, 3.12, or 3.13
- Core Libraries: `opencv-python` (HOG extraction), `scikit-learn` (SVM, scaling, metrics), `numpy` (array processing), `matplotlib` & `seaborn` (visualization), `pandas` (table rendering)

### Data Specifications
- Medical Benchmark: PneumoniaMNIST
- Class Labels: Normal Lung, Bacterial Pneumonia, Viral Pneumonia
- Image Resolution: 128 x 128 pixels
- Color Channels: Single-channel grayscale (local anatomical contrast normalized)
- Dataset Partition: 60% Training, 20% Validation, 20% Held-out Test
- Data Split Method: Identical fixed random split using Seed 42 across baseline and proposed models

### Feature Extraction Parameters
- Detection Window: 128 x 128
- Block Size: 16 x 16 (2 x 2 cells)
- Block Stride: 16 x 16 (non-overlapping blocks across the image grid)
- Cell Size: 8 x 8 pixels
- Orientation Bins: 9 bins (0 to 180 degrees)
- Block Normalization: L2-Hys (L2-norm with clipping at 0.2 and re-normalization)
- Full Feature Dimension: 64 blocks x 36 values = 2,304 features per image

### Genetic Algorithm Specifications
- Representation: Binary chromosome of length 2,304 (1 = feature kept, 0 = feature dropped)
- Population Size: Evaluated with 30 (Setting 1) and 50 (Setting 2)
- Mutation Rate: 0.04 (Setting 1) vs 0.08 (Setting 2)
- Crossover Method: Uniform vs Two-Point
- Generations: 20 generations with 2-individual elitism preservation
- Fitness Function: Validation Macro-F1 combined with a linear sparsity penalty:
  $$\text{Fitness} = \alpha \cdot \text{Macro-F1}_{\text{val}} + (1 - \alpha) \cdot \left(1 - \frac{\sum \text{gene}_i}{2304}\right)$$

---

## 2. Research Question & Summary of Findings

> **Research Question:** Can Genetic Algorithm-based feature selection improve the classification performance of HOG features while reducing feature dimensionality on medical images?

### Answer
Yes. Dense HOG descriptors generate 2,304 features per 128 x 128 medical image. Many of these features capture empty image borders, background illumination shifts, or noise that has no diagnostic value.

By applying a Genetic Algorithm driven by validation Macro-F1 and a sparsity penalty:
1. **Feature Reduction:** Feature dimensionality dropped from **2304** down to **445** features (**80.69% reduction**).
2. **Classification Quality:** Test Macro-F1 rose from **94.08%** to **98.01%** (**+3.93% improvement**), and Accuracy increased from **94.12%** to **98.04%** (**+3.92%**).
3. **Inference Latency:** Average inference latency dropped from **0.204 ms** to **0.041 ms**, achieving a **4.98x speedup** per image.

---

## 3. End-to-End System Architecture

The complete workflow from raw input image to final diagnostic classification is shown below:

<p align="center">
  <img src="figures/01_pipeline_architecture.png" alt="System Architecture Pipeline" width="950">
</p>

1. **Input Preprocessing:** Each medical image is resized to 128 x 128 and converted to standardized grayscale.
2. **HOG Feature Extraction:** Local cell gradient histograms are computed across 9 bins and normalized over 16 x 16 blocks, producing a 2,304-dimensional vector.
3. **Evolutionary Feature Selection:** The Genetic Algorithm iteratively evaluates feature subsets on the validation split over 20 generations, rewarding both classification quality and feature sparsity.
4. **Subspace Transformation:** The optimal chromosome masks the feature space, keeping only the 445 most informative gradient features.
5. **RBF SVM Classification:** A non-linear Support Vector Machine with Radial Basis Function kernel classifies the test sample.

---

## 4. Experimental Configuration (Table 1)

This table summarizes the experimental parameters so that all results can be reproduced:

| Parameter | Baseline: HOG + SVM | Proposed: GA-HOG + SVM |
| :--- | :--- | :--- |
| **Medical Dataset** | PneumoniaMNIST | PneumoniaMNIST |
| **Image Resolution** | 128 x 128 | 128 x 128 |
| **HOG Orientations** | 9 | 9 |
| **Pixels per Cell** | 8 x 8 | 8 x 8 |
| **Cells per Block** | 2 x 2 | 2 x 2 |
| **Block Normalization** | L2-Hys | L2-Hys |
| **Feature Selection** | None (All 2,304 features used) | **Genetic Algorithm (Setting 1 (Conservative GA: Pop=30, Mut=0.04))** |
| **Classifier** | Support Vector Machine (SVM) | Support Vector Machine (SVM) |
| **SVM Kernel** | RBF (C = 2.0, gamma = 'scale') | RBF (C = 2.0, gamma = 'scale') |
| **Hyperparameter Tuning** | Evaluated on validation set | Evaluated on validation set |
| **Data Split** | Fixed 60% Train, 20% Val, 20% Test | Fixed 60% Train, 20% Val, 20% Test |
| **Random Seed** | Fixed (42) | Fixed (42) |

---

## 5. Main Performance Comparison (Table 2)

All evaluation metrics below were computed on the held-out test set after model selection was completed. Specificity is calculated using a multiclass macro-averaged one-vs-rest formulation:

$$\text{Specificity}_{\text{Macro}} = \frac{1}{C} \sum_{c=1}^C \frac{TN_c}{TN_c + FP_c}$$

| Metric | HOG + SVM (Baseline) | GA-HOG + SVM (Proposed) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy (%)** | **94.12%** | **98.04%** | **+3.92%** |
| **Precision (Macro, %)** | **94.35%** | **98.15%** | **+3.8%** |
| **Recall (Macro, %)** | **94.12%** | **98.04%** | **+3.92%** |
| **Macro-F1 (%)** | **94.08%** | **98.01%** | **+3.93%** |
| **Weighted-F1 (%)** | **94.1%** | **98.03%** | **+3.93%** |
| **Balanced Accuracy (%)** | **94.12%** | **98.04%** | **+3.92%** |
| **Specificity (Macro, %)** | **97.06%** | **99.02%** | **+1.96%** |
| **Cohen's Kappa** | **0.9118** | **0.9706** | **+0.0588** |
| **Matthews Correlation Coefficient (MCC)** | **0.9125** | **0.9708** | **+0.0583** |
| **Feature Dimensionality** | 2304 features | 445 features | **-80.69% reduction** |
| **Inference Latency per Sample (ms)** | 0.204 ms | 0.041 ms | **4.98x speedup** |

---

## 6. Visual Walkthrough of Experimental Results

### 1. Medical Samples and HOG Gradient Responses
The figure below displays sample images from each medical category alongside their corresponding Sobel gradient magnitudes. The gradient field highlights anatomical boundaries and pathological lesions that the HOG descriptors capture.

<p align="center">
  <img src="figures/02_medical_samples_hog.png" alt="Medical Samples and HOG Vector Magnitude" width="850">
</p>

---

### 2. Genetic Algorithm Convergence Across Generations
We compared two GA configurations to evaluate population size and mutation rate dynamics. The graph below displays best and mean fitness curves across 20 generations.

<p align="center">
  <img src="figures/03_ga_fitness_convergence.png" alt="GA Fitness Convergence" width="750">
</p>

- **Setting 1 (Setting 1 (Conservative GA: Pop=30, Mut=0.04)):** Best validation fitness reached 0.9412 with 445 active features.
- **Setting 2 (Setting 2 (Exploratory GA: Pop=50, Mut=0.08)):** Best validation fitness reached 0.9285 with 512 active features.

---

### 3. Feature Space Pruning Distribution
The bar and donut chart below illustrate the scale of feature reduction achieved by the Genetic Algorithm:

<p align="center">
  <img src="figures/04_feature_reduction_distribution.png" alt="Feature Reduction Distribution" width="800">
</p>

The algorithm discarded 1859 features (80.69%), leaving only 445 features that provided the highest discriminative power.

---

### 4. Confusion Matrix Comparison (Test Set)
Side-by-side confusion matrices show how feature selection improved classification by eliminating false positives and false negatives on the unseen test split:

<p align="center">
  <img src="figures/05_confusion_matrices.png" alt="Confusion Matrix Comparison" width="850">
</p>

---

### 5. Overall Clinical Metrics Comparison
The grouped bar chart below compares all evaluated performance metrics between the baseline HOG pipeline and the proposed GA-HOG pipeline:

<p align="center">
  <img src="figures/06_metrics_radar_comparison.png" alt="Metrics Comparison" width="850">
</p>

---

## 7. Practical Discussion

### Why GA Feature Selection Improves Medical Classification
Standard HOG descriptors calculate gradients systematically across every grid cell in the image. In medical images, a large fraction of the image is made up of background tissue or scanner artifacts that contain zero diagnostic information. When the SVM is trained on all 2,304 features, these noisy dimensions dilute the kernel dot product and lead to sub-optimal decision boundaries.

The Genetic Algorithm filters out this noise by testing subsets on validation data. By keeping only the 445 most relevant gradient bins, the classifier focuses directly on key morphological boundaries, resulting in higher balanced accuracy and macro-F1.

### Trade-offs and Computational Costs
- **Offline Training Time:** The main drawback of GA feature selection is the training overhead. Evaluating a population of 30 to 50 individuals over 20 generations requires hundreds of validation passes. However, this is an offline training cost that only needs to be run once.
- **Inference Speedup:** Once model selection is done, the selected feature mask makes the deployed system much faster. Because the model computes dot products over only 445 features instead of 2,304, per-image latency drops from 0.204 ms to 0.041 ms (4.98x speedup). This makes the model well-suited for edge clinical devices.

---

## 8. How to Run the Project

### Option A: Open Directly in Google Colab (Recommended)
You do not need to install anything locally. Click the badge below to run the complete pipeline directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheBilalButt/lab-07-ga-hog-medical-classification/blob/main/Lab07_GA_HOG_Medical_Classification.ipynb)

### Option B: Run Locally via Python CLI
1. Clone the repository:
   ```bash
   git clone https://github.com/TheBilalButt/lab-07-ga-hog-medical-classification.git
   cd lab-07-ga-hog-medical-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the standalone pipeline script:
   ```bash
   python ga_hog_classification.py
   ```
4. Or launch the Jupyter Notebook:
   ```bash
   jupyter notebook Lab07_GA_HOG_Medical_Classification.ipynb
   ```

---

## 9. Author Information
- **Author:** Muhammad Bilal Butt
- **Student ID:** FA23-BAI-042
- **Course:** Computer Vision (Lab 07)
- **Department:** Computer Science / Artificial Intelligence
