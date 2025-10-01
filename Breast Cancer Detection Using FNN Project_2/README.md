#  Breast Cancer Detection Using Feedforward Neural Network (FNN)

## Project Overview
Breast cancer remains one of the leading causes of death among women worldwide. Early detection drastically improves survival rates, yet traditional methods can be invasive, time-consuming, and prone to human error.  

This project applies **Deep Learning (Feedforward Neural Network - FNN)** to classify breast tumors as either **Malignant (M)** or **Benign (B)** based on digitized features of cell nuclei. The model leverages structured data to provide fast, accurate, and reliable predictions that can assist radiologists and oncologists in clinical decision-making.  

---

## Problem Statement
- **Challenge:** Distinguish between malignant and benign breast tumors with high accuracy.  
- **Impact:** Early and precise classification can reduce unnecessary biopsies, optimize resource allocation, and improve patient survival rates.  
- **Goal:** Build a **Deep Learning-based FNN classifier** that can achieve **>90% accuracy** in predicting breast cancer status.  

---

## Business Insights & Use Cases
1. **Early Diagnosis** → Helps doctors identify malignant tumors early, reducing treatment delays.  
2. **Resource Allocation** → Supports healthcare providers in prioritizing high-risk patients.  
3. **Cost Reduction** → Minimizes invasive procedures by serving as a diagnostic support tool.  
4. **Public Health Impact** → Enhances preventive screening programs, lowering long-term treatment costs.  

---

## Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Shape:** `569 samples × 31 features`  
- **Target Variable:** `Diagnosis` → Malignant (M=1), Benign (B=0)  
- **Key Features Used (strong correlation with malignancy):**
  - `concave points_worst` (0.79)  
  - `perimeter_worst` (0.78)  
  - `concave points_mean` (0.77)  
  - `radius_worst` (0.77)  
  - `perimeter_mean` (0.74)  
  - `area_worst` (0.73)  
  - `radius_mean` (0.73)  
  - `area_mean` (0.71)  

- **Dropped Features:** `id`, `Unnamed: 32` (irrelevant or empty).  

---

## Data Preprocessing
- Removed irrelevant columns (`id`, `Unnamed: 32`).  
- Checked for missing values → None found.  
- Label encoded target (`diagnosis`: M=1, B=0).  
- Handled outliers using **IQR method** (kept as dataset is balanced & clean).  
- Standardized features using **StandardScaler**.  
- Split dataset: **80% training, 20% testing**.  

---

## Tech Stack
- **Languages:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
- **Deep Learning Frameworks:** TensorFlow, Keras  
- **Environment:** Google Colab  

---

## Model Architecture (Feedforward Neural Network)
- **Input Layer:** 30 features  
- **Hidden Layer 1:** Dense(16), ReLU activation  
- **Hidden Layer 2:** Dense(8), ReLU activation  
- **Output Layer:** Dense(1), Sigmoid activation  

**Optimizer:** Adam  
**Loss Function:** Binary Crossentropy  
**Batch Size:** 10  
**Epochs:** 100  

---

## Results
- **Accuracy:** `94.73%`  
- **Classification Report:**  
  - Precision (Malignant): `0.89`  
  - Recall (Malignant): `1.00`  
  - F1-Score (Malignant): `0.94`  
  - Weighted Avg F1-Score: `0.95`  

- **Confusion Matrix:**  
[[61 6]
[ 0 47]]
- Correctly classified benign tumors: 61  
- Correctly classified malignant tumors: 47  

- The model achieved **strong performance**, particularly excelling at minimizing false negatives (critical in medical diagnosis).  

---

## Future Improvements
1. **Hyperparameter Tuning** → Use GridSearch/RandomSearch for optimizing neurons, layers, and learning rate.  
2. **Regularization** → Add Dropout/L2 Regularization to further prevent overfitting.  
3. **Cross-Validation** → Improve robustness by validating across multiple folds.  
4. **Explainability (XAI)** → Use SHAP/LIME for feature attribution and model transparency.  
5. **Deployment** → Wrap into a web or mobile app for doctors to use in real-time.  
6. **Integration** → Connect with radiology imaging data for a hybrid image + structured data model.  

---

## Conclusion
This project demonstrates how **Deep Learning (FNN)** can be applied effectively to **breast cancer detection**, achieving **~95% accuracy**. By identifying high-risk patients early and reducing diagnostic overhead, this model shows promise as a **decision-support tool in modern healthcare**.  

Future advancements such as **explainability, deployment, and multimodal integration** can transform this proof-of-concept into a real-world clinical assistant, ultimately saving lives and reducing healthcare costs.  

---















