# Predicting Real Estate Unit Prices Using Deep Learning

## Project Overview
Accurate real estate pricing is critical for **buyers, sellers, and investors** in competitive housing markets.  
This project focuses on predicting **residential property prices per unit area** in **Taipei, Taiwan** by leveraging **Deep Learning (Feedforward Neural Network)**.  

By analyzing **historical transaction data** and **property features** such as age, proximity to MRT stations, number of convenience stores, and geolocation, this project aims to estimate **fair and data-driven property values**.  

---

## Problem Statement
- **Challenge:** Property prices fluctuate due to multiple interdependent factors. Traditional valuation methods often fail to capture these relationships.  
- **Objective:** Build a **Deep Learning model** that predicts property unit prices with high accuracy, and compare performance against a **baseline ML model (Linear Regression)**.  
- **Impact:**  
  - Helps **buyers** avoid overpriced properties.  
  - Assists **sellers/investors** in setting fair prices.  
  - Provides **urban planners** and **real estate firms** with data-driven insights.  

---

## Business Insights from Dataset
1. **Age vs. Price** → Older houses generally have lower value.  
2. **Proximity to MRT Station** → Properties closer to MRT stations tend to be more expensive, highlighting the role of **transport accessibility**.  
3. **Convenience Stores** → More nearby stores correlate slightly with higher unit prices, reflecting **urban infrastructure quality**.  
4. **Location (Lat/Long)** → Geographic differences strongly influence pricing patterns.  

---

## Dataset
- **Source:** Real Estate Valuation Dataset (Taiwan)  
- **Shape:** `414 samples × 7 features`  
- **Target Variable:** `Y house price of unit area`  
- **Features:**
  - `X1 transaction date` – Date of property sale  
  - `X2 house age` – Age of the property (years)  
  - `X3 distance to nearest MRT station` (meters)  
  - `X4 number of convenience stores` nearby  
  - `X5 latitude`, `X6 longitude` – Location coordinates  
- **Removed Feature:** `No` (index column, irrelevant).  

---

## Data Preprocessing
- Dropped irrelevant column (`No`).  
- Checked for missing values → None found.  
- Verified duplicates → None present.  
- Outlier analysis → Dataset clean, small impact of outliers.  
- Standardized features with **StandardScaler** for better model convergence.  
- Train-test split: **80% training, 20% testing**.  

---

## Tech Stack
- **Languages:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
- **Deep Learning Frameworks:** TensorFlow, Keras  
- **Environment:** Google Colab  

---

## Model Architectures

### Deep Learning Model (FNN)
- **Input Layer:** 6 features  
- **Hidden Layer 1:** Dense(10), ReLU activation  
- **Hidden Layer 2:** Dense(8), ReLU activation  
- **Output Layer:** Dense(1), Linear activation (for regression)  
- **Optimizer:** Adam  
- **Loss Function:** Mean Squared Error (MSE)  
- **Epochs:** 100, Batch size = 10  

### Baseline Model (Machine Learning)
- Multiple Linear Regression (Scikit-learn)  

---

## Results

### FNN Model (Keras/TensorFlow)
- **Mean Squared Error (MSE):** `42.60`  
- **R² Score:** `0.75`  
- Captures **non-linear relationships** better than regression baseline.  

### Linear Regression Model
- **Mean Squared Error (MSE):** `59.52`  
- Lower accuracy compared to Deep Learning model.  

### Visualization
- Scatter plots show **predicted vs. actual property prices** for both models.  
- FNN predictions are closer to true values compared to Linear Regression.  

---

## Future Improvements
1. **Hyperparameter Tuning** → Experiment with neurons, layers, learning rates using GridSearch or Bayesian optimization.  
2. **Regularization** → Apply **Dropout** or **L2 Regularization** to reduce overfitting.  
3. **Cross-Validation** → Perform K-Fold CV to ensure robustness.  
4. **Feature Engineering** → Add more socio-economic or environmental features (e.g., school quality, air quality, economic index).  
5. **Ensemble Learning** → Combine Deep Learning with XGBoost or Random Forest to boost performance.  
6. **Deployment** → Deploy as a **real estate pricing API** or integrate into property listing platforms.  

---

## Conclusion
This project demonstrates the **power of Deep Learning** in predicting **real estate unit prices**.  
The **Feedforward Neural Network** outperforms traditional **Linear Regression**, achieving an **R² of 0.75**, indicating strong predictive capability.  

While the model is promising, future enhancements such as **feature engineering, explainability (SHAP/LIME), and deployment** can transform it into a **real-world decision-support system** for the real estate industry.  

---

