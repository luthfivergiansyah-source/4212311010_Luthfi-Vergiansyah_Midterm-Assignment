# 4212311010_Luthfi-Vergiansyah_Midterm-Assignment
This midterm exam implements a handwritten letter recognition system using the EMNIST Letters dataset.
The goal is to classify handwritten alphabet characters (A–Z) using Histogram of Oriented Gradients (HOG) for feature extraction and Support Vector Machine (SVM) for classification.
The model is evaluated using Leave-One-Out Cross Validation (LOOCV) to ensure high reliability and accuracy.

Dataset Loading
The program loads the EMNIST Letters dataset (.ubyte format) containing grayscale images of handwritten letters.
Each image is reshaped and normalized for preprocessing.

Data Sampling
A balanced subset is created by selecting 500 samples per class, resulting in 13,000 total samples.
This ensures each alphabet letter is equally represented during training.

Feature Extraction (HOG)
Orientations: 9
Pixels per Cell: 8×8
Cells per Block: 2×2
Block Normalization: L2-Hys
These parameters help capture gradient and edge direction patterns that define the shape of each letter.

Model Training (SVM)
A Support Vector Machine (SVM) classifier with RBF kernel is used.
The parameters are set as C=10 and gamma=0.01, which balance generalization and boundary precision.

Model Evaluation (LOOCV)
The Leave-One-Out Cross Validation (LOOCV) method is applied.
Every single sample is tested once while the remaining samples are used for training, producing an unbiased performance estimate.

Result Visualization
A confusion matrix is generated and visualized with seaborn.heatmap to analyze the model’s classification performance for each letter.

Conclusion
The combination of HOG for feature extraction and SVM (RBF kernel) for classification proves to be effective in recognizing handwritten letters.
The model demonstrates strong accuracy and robustness, making it a reliable approach for offline handwriting recognition tasks.

Author
Luthfi Vergiansyah
4212311010
Mechatronics 5 Night
