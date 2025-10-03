# Heart Attack Risk Prediction Using Deep Learning  
*A Classification Approach with Feedforward Neural Networks (FNN)*  

---

## Project Overview  
Cardiovascular diseases remain one of the leading causes of death globally. Early identification of heart attack risk can significantly enhance **preventive care, reduce costs, and save lives**.  

In this project, we developed a **Deep Learning-based classification model (Feedforward Neural Network)** that predicts **heart attack risk** based on patient health records. The model classifies patients into two categories:  
- **0 → Low Risk**  
- **1 → High Risk**  

Our workflow includes **data preprocessing, feature engineering, scaling, balancing, feature selection, and model development** using **TensorFlow/Keras**.  

---

##  Problem Statement  
The aim is to:  
- Identify the **key health risk factors** influencing heart attacks.  
- Build a **predictive deep learning model** that can generalize well.  
- Assist **healthcare professionals** in early diagnosis and decision-making.  

---

## Business Insights  
1. **Healthcare Cost Reduction** → Preventing severe cardiac cases reduces hospital costs.  
2. **Preventive Care** → High-risk individuals can adopt lifestyle interventions earlier.  
3. **Resource Allocation** → Hospitals can prioritize patients based on predicted risk.  
4. **Insurance Optimization** → Risk-based premium models can be implemented.  

---

## Dataset Details  
- **Total Samples:** 8,763  
- **Features:** 25 → reduced to 21 after preprocessing  
- **Target Variable:** `Heart Attack Risk` (0 = Low Risk, 1 = High Risk)  

### Key Features Considered  
- **Demographics:** Age, Sex, Family History  
- **Health Indicators:** Cholesterol, Blood Pressure, Heart Rate, BMI  
- **Lifestyle:** Smoking, Alcohol Consumption, Diet, Physical Activity  
- **Medical Records:** Diabetes, Previous Heart Problems, Medication Use, Stress Level  

### Dropped Features  
- `Patient ID`, `Country`, `Continent`, `Hemisphere`, `Income` → Not predictive or indirect  

---

## Data Preprocessing  
- **Missing Values** → None found  
- **Duplicates** → Removed  
- **Categorical Encoding**:  
  - `Sex` → Label Encoding  
  - `Blood Pressure` → Split into `Systolic_BP` & `Diastolic_BP`  
  - `Diet` → Ordinal Encoding (Unhealthy < Average < Healthy)  
- **Balancing** → Applied **SMOTE** to handle class imbalance  
- **Scaling** → StandardScaler used for feature normalization  
- **Feature Selection** → ANOVA F-test, Correlation, Mutual Information → selected top predictors like:  
  - `Previous Heart Problems`, `Family History`, `Alcohol Consumption`, `Medication Use`, `Stress Level`, `Obesity`, `Diet`  

---

## Model Development  

### **Architecture**  
- Input Layer → 21 features  
- Dense Layer (32 neurons, ReLU, L2 regularization)  
- Dropout (0.3)  
- Dense Layer (16 neurons, ReLU, L2 regularization)  
- Dropout (0.3)  
- Output Layer (1 neuron, Sigmoid) → Binary Classification  

### **Compilation**  
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Metrics:** Accuracy  

### **Training Setup**  
- **Batch Size:** 10  
- **Epochs:** 100 (with Early Stopping, patience = 5)  
- **Validation Split:** 20%  

---

## Model Performance  

- **Train Accuracy:** 63.71%  
- **Test Accuracy:** 62.67%  

### Classification Report  
| Class | Precision | Recall | F1-score | Support |  
|-------|------------|--------|-----------|---------|  
| 0 (Low Risk) | 0.60 | 0.74 | 0.66 | 1122 |  
| 1 (High Risk) | 0.67 | 0.51 | 0.58 | 1128 |  

- **Confusion Matrix:**  
[[833 289]
[551 577]]

### Insights  
- Model shows **moderate accuracy (~63%)**.  
- Performs **better on low-risk prediction** compared to high-risk cases.  
- Indicates potential underfitting due to model simplicity and data complexity.  

---

## Tech Stack  
- **Programming Language**: Python 3.x  
- **Libraries**:  
- Data Processing → Pandas, NumPy  
- Visualization → Matplotlib, Seaborn  
- Machine Learning (Preprocessing/Feature Selection) → Scikit-learn, Imbalanced-learn  
- Deep Learning → TensorFlow, Keras  
- **Environment**: Google Colab / Jupyter Notebook  

---

## Future Improvements  
1. **Model Enhancements**  
 - Try deeper neural networks or advanced architectures (e.g., CNN on health signal data, LSTM for time-series health records).  
 - Experiment with hyperparameter tuning (batch size, learning rate, regularization).  
 - Use ensemble approaches combining ML + DL.  

2. **Data Improvements**  
 - Collect more diverse patient data across regions.  
 - Include lab test results (ECG, echocardiograms, etc.) for richer feature space.  
 - Perform feature engineering with interaction terms (e.g., Age × BMI).  

3. **Deployment**  
 - Deploy model as a **web app or API** for hospitals/clinics.  
 - Integrate with wearable devices for real-time monitoring.  

---

## Conclusion  
This project successfully built a **deep learning-based predictive model** for heart attack risk classification. While the achieved accuracy (~63%) indicates room for improvement, the workflow demonstrates the power of **data preprocessing, feature engineering, and neural networks** in healthcare analytics.  

With further improvements in **data quality, model complexity, and real-world validation**, this approach can evolve into a **decision-support system for doctors**, ultimately saving lives through early detection.  

---


















