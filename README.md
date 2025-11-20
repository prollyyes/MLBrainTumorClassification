# MLBrainTumorClassification
Using CNNs and SVMs to classify MRI Brain Scans whether or not the patient has a brain tumor, and which kind.

***

## Overview

This project investigates the effectiveness of combining pretrained convolutional neural network representations with classical machine learning models for multi-class brain tumor classification. EfficientNetB0 is used as a frozen feature extractor to obtain high-dimensional embeddings, which are then classified using traditional models such as Logistic Regression and Support Vector Machines (SVM). The study evaluates model performance, generalization behavior, and class separability through a comprehensive set of quantitative and qualitative analyses.

## Dataset

The dataset contains four MRI categories:
	•	Glioma tumor
	•	Meningioma tumor
	•	Pituitary tumor
	•	No tumor

All images are processed into 1280-dimensional feature vectors via EfficientNetB0.

# Methodology

## Feature Extraction
	•	EfficientNetB0 (pretrained on ImageNet) is used as a fixed feature extractor.
	•	Embeddings are standardized using StandardScaler.

## Models Evaluated
	•	Gaussian Naive Bayes
	•	Logistic Regression (baseline)
	•	SVM with RBF kernel (primary model)
	•	Decision Tree
	•	Random Forest

*Logistic Regression is used as a consistent linear baseline; the SVM serves as the main nonlinear model.*

## Hyperparameter Tuning
	•	SVM hyperparameters (C, γ) are tuned using 5-fold GridSearchCV.
	•	The best model is retrained on the combined training and validation sets.

## Evaluation Methods
	•	Accuracy, precision, recall, F1-score
	•	Confusion matrices
	•	One-vs-Rest ROC curves and AUC
	•	Learning curves
	•	PCA-based 2D visualizations
	•	Feature importance via Logistic Regression coefficients (SVM does not provide meaningful feature importances due to the implicit RBF feature space)

# Results

## Model Performance
	•	SVM (RBF) achieves the highest performance with test accuracy ≈ 0.93 and strong per-class metrics.
	•	Logistic Regression reaches ≈ 0.91 test accuracy, demonstrating the strength of the embedding space.
	•	Other classical models perform reasonably but below SVM and Logistic Regression.

## Confusion Matrix Observations
	•	Strong diagonal structure for SVM, indicating reliable predictions.
	•	Main confusions occur between glioma and meningioma, which is consistent with their visual similarity.
	•	No-tumor and pituitary classes exhibit near-perfect separation.

## ROC and AUC
	•	One-vs-Rest ROC curves show high separability for all classes.
	•	AUC values are consistently high; the no-tumor class achieves AUC = 1.00.

## Learning Curves
	•	SVM displays mild overfitting but benefits significantly from additional data.
	•	Logistic Regression shows low variance and stable convergence.

## Feature Importance
	•	SVM with RBF kernel cannot provide feature-level importance.
	•	Logistic Regression coefficients indicate that discriminative information is distributed across many embedding dimensions.

# Conclusion

The combination of EfficientNet-based representations with classical machine learning models yields strong performance on multi-class brain tumor classification. The results demonstrate that high-quality embeddings can enable simple models such as SVMs and Logistic Regression to achieve robust accuracy, clear separability, and reliable generalization. The study provides a solid baseline for future work involving fine-tuning, broader datasets, or enhanced interpretability methods.
