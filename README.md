# 🌭 Hotdog vs. Not Hotdog: Binary Image Classification

This project implements an automated food recognition system using **Deep Learning** to differentiate between hotdog images and non-hotdog images. It compares a custom-built **Convolutional Neural Network (CNN)** against a **Transfer Learning (VGG16)** approach.

---

## 1. Problem Domain & Applications
Retail and food industries require vast datasets to automate visual recognition. This binary classification model addresses:
* **Smart restaurant ordering systems**.
* **Automated food recognition** in retail and self-checkout kiosks.
* **AI-based image tagging** applications.

**Class Labels:**
* **Class 0:** Hotdog.
* **Class 1:** Not Hotdog.

---

## 2. Data Engineering & Cleaning
To ensure high-quality training, the following data cleaning steps were performed as seen in the Python script:

* **Corrupted Image Removal:** Used the Python **PIL** library to `verify()` images, automatically deleting damaged or unreadable files that could cause training errors.
* **Duplicate Detection:** Applied **MD5 hashing** to identify and remove duplicate images, preventing model bias and overfitting.
* **Dataset Splitting:** The dataset was programmatically split into **70% Training**, **15% Validation**, and **15% Testing** sets.

### Exploratory Data Analysis (EDA)
Before training, we analyzed class distributions and pixel characteristics to ensure a balanced dataset.




* **Image Dimensions:** A scatter plot was used to analyze varying widths and heights before resizing.
* **Pixel Intensity:** KDE plots compared the brightness distributions between classes.



---

## 3. Image Preprocessing
Deep learning models require consistent inputs for efficient weight updates.
* **Resizing:** All images were resized to $150 \times 150$ pixels.
* **Normalization:** Pixel values were rescaled to a range of $[0, 1]$ to improve training stability and convergence speed.
* **Augmentation:** Applied random rotations ($20^\circ$), zooms ($0.2$), and horizontal flips to the training set to improve generalization.

---

## 4. Modeling Approach
We compared a baseline custom architecture against a state-of-the-art transfer learning model.

### 4.1 Custom CNN Model
Designed for speed and lower computational costs.
* **Architecture:** 3 Convolutional layers (32, 64, 128 filters) with **BatchNormalization** and **MaxPooling**.
* **Classifier:** A Dense layer with 128 neurons and **Dropout (0.5)** to reduce overfitting.
* **Output:** A single neuron with a **Sigmoid** activation function for binary classification.



### 4.2 VGG16 (Transfer Learning)
Leverages pre-trained weights from the ImageNet dataset.
* **Base:** Frozen VGG16 convolutional base to retain learned visual features (edges, textures).
* **Top:** Custom classifier including **GlobalAveragePooling2D**, a Dense layer (256 neurons), and Dropout (0.5).

---

## 5. Performance Evaluation
The models were evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

### Training Metrics
We tracked the performance over multiple epochs to identify potential overfitting.



### Results Comparison
The performance was quantified using the **Confusion Matrix** and **ROC Curve**.

* **Confusion Matrix:** Visualizes True Positives vs. False Positives to identify where the model is confused.
* **ROC Curve:** Measures the model's ability to differentiate between the two classes. A higher **AUC (Area Under the Curve)** indicates a more robust model.




### Key Outcomes
* **VGG16 Superiority:** The transfer learning model generally provided higher accuracy and better generalization than the custom CNN.
* **Feature Learning:** CNN layers successfully learned hierarchical features, from simple edges to complex shapes like the curve of a bun.
  
  **Feature Map Calculation:**
  $$Feature Map = (Input \times Filter) + Bias$$

---

## 6. Conclusion & Future Work
The project demonstrates that deep learning can effectively automate food recognition tasks. While both models performed well, **VGG16** is the preferred choice for production due to its high precision.

**Future Improvements:**
* Implement **MobileNet** or **EfficientNet** for faster inference on mobile devices.
* Expand the dataset with more diverse backgrounds and lighting conditions.
* Fine-tune additional layers of the VGG16 model to improve accuracy.

---
### 🛠️ Tech Stack
* **Language:** Python
* **Frameworks:** TensorFlow, Keras
* **Libraries:** NumPy, Matplotlib, Seaborn, PIL, Scikit-learn
