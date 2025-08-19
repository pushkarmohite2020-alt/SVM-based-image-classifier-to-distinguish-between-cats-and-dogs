# SVM-based-image-classifier-to-distinguish-between-cats-and-dogs
✔ Trained a Linear SVM classifier from scratch ✔ Preprocessed images (resizing, grayscale conversion) for feature extraction ✔ Achieved classification metrics (accuracy, precision, recall) ✔ Visualized predictions with sample test images 📊 Results: Accuracy: 66% (with room for improvement!) Explored challenges in raw pixel-based classification
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from google.colab import files
from skimage import exposure

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Loading and Preparation
# --------------------------------------------------
print("1. Loading and preparing dataset...")

# Unzip dataset (assuming you've uploaded train.zip)
# Add a check if the directory already exists to avoid errors on re-running
if not os.path.exists('dataset'):
    !unzip -q train.zip -d dataset
else:
    print("Dataset directory already exists. Skipping unzip.")

def load_and_preprocess_data(sample_size=2000):
    """Load images and extract HOG features"""
    X = []
    y = []
    data_dir = '/content/dataset/train'

    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found at {data_dir}")
        return np.array(X), np.array(y)

    filenames = os.listdir(data_dir)

    # Check if there are any files in the directory
    if not filenames:
        print(f"Error: No files found in directory {data_dir}")
        return np.array(X), np.array(y)

    # Limit the number of files to process
    filenames = filenames[:sample_size]

    for filename in tqdm(filenames, desc="Processing Images"):
        # Label: 1 for dog, 0 for cat
        label = 1 if 'dog' in filename.lower() else 0
        img_path = os.path.join(data_dir, filename)

        try:
            # Read and resize image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # print(f"Warning: Could not read image {filename}")
                continue

            img = cv2.resize(img, (128, 128))

            # Extract HOG features
            features = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )

            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    return np.array(X), np.array(y)

# Load dataset
X, y = load_and_preprocess_data(sample_size=2000)  # Using 2000 images for faster training

# Check if any data was loaded
if X.shape[0] == 0:
    print("No data loaded. Please check the dataset path and file names.")
else:
    # 2. Data Preprocessing
    # --------------------------------------------------
    print("\n2. Preprocessing data...")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Check if train and test sets are not empty after splitting
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Error: Training or test set is empty after splitting. This might happen if the sample size is too small or data loading failed.")
    else:
        print(f"\nDataset Summary:")
        print(f"- Training samples: {X_train.shape[0]}")
        print(f"- Test samples: {X_test.shape[0]}")
        print(f"- Number of features: {X_train.shape[1]}")
        print(f"- Class distribution (Train): Cats={np.sum(y_train==0)}, Dogs={np.sum(y_train==1)}")
        print(f"- Class distribution (Test): Cats={np.sum(y_test==0)}, Dogs={np.sum(y_test==1)}")

        # 3. Model Training (SVM)
        # --------------------------------------------------
        print("\n3. Training SVM classifier...")

        # Initialize SVM with linear kernel
        svm = SVC(
            kernel='linear',  # Linear kernel for better interpretability
            C=1.0,           # Regularization parameter
            random_state=42,
            verbose=True     # Show training progress
        )

        # Train the model
        svm.fit(X_train, y_train)

        # 4. Model Evaluation
        # --------------------------------------------------
        print("\n4. Evaluating model performance...")

        # Make predictions
        y_pred = svm.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\nPerformance Metrics:")
        print(f"- Accuracy: {accuracy:.2%}")
        print(f"- Precision: {precision:.2%}")  # True positives / (True positives + False positives)
        print(f"- Recall: {recall:.2%}")       # True positives / (True positives + False negatives)
        print(f"- F1 Score: {f1:.2%}")         # Harmonic mean of precision and recall

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['Cat', 'Dog'])
        plt.yticks(tick_marks, ['Cat', 'Dog'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")

        plt.tight_layout()
        plt.show()

        # 5. Visualization of Sample Predictions
        # --------------------------------------------------
        print("\n5. Visualizing sample predictions...")

        def visualize_prediction(img_path, model, scaler):
            """Visualize image with prediction"""
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image for visualization: {img_path}")
                return

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract features and predict
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))
            features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )
            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)[0]

            # Display
            plt.figure(figsize=(4, 4))
            plt.imshow(img_rgb)
            plt.title(f"Prediction: {'Dog' if pred == 1 else 'Cat'}")
            plt.axis('off')
            plt.show()

        # Test on sample images
        sample_images = [
            '/content/dataset/train/cat.100.jpg',
            '/content/dataset/train/dog.100.jpg',
            '/content/dataset/train/cat.200.jpg',
            '/content/dataset/train/dog.200.jpg'
        ]

        for img_path in sample_images:
            if os.path.exists(img_path):
                visualize_prediction(img_path, svm, scaler)
            else:
                print(f"Image not found for visualization: {img_path}")

        # 6. Save Model (Optional)
        # --------------------------------------------------
        # Uncomment to save the trained model
        # import joblib
        # joblib.dump(svm, 'dogs_vs_cats_svm.pkl')
        # print("Model saved as 'dogs_vs_cats_svm.pkl'")
