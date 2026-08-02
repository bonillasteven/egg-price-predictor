# 🥚 Egg Price Predictor

A machine learning project that predicts the monthly average retail price of one dozen Grade A large eggs in the United States using historical economic and agricultural data.

---

## 📖 Project Overview

Egg prices are influenced by many factors, including feed costs, fuel prices, inflation, and disease outbreaks affecting poultry. This project combines multiple public datasets and applies machine learning techniques to predict monthly egg prices.

The project follows a complete machine learning workflow, including:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Feature Importance Analysis
- Time Series Validation
- Interactive Prediction Program

---

## 🎯 Project Objective

The objective of this project is to build a machine learning model capable of accurately predicting the monthly average retail price of one dozen Grade A large eggs in the United States.

---

## 📊 Dataset

This project combines several publicly available datasets.

| Dataset | Description |
|---------|-------------|
| Egg Prices | Monthly average retail egg prices |
| Corn Prices | Feed cost indicator |
| Soybean Prices | Feed cost indicator |
| Diesel Prices | Transportation cost indicator |
| Consumer Price Index (CPI) | Economic indicator |
| Inflation Rate | Engineered feature from CPI |
| Bird Flu Data | Number of birds affected by outbreaks |
| Bird Flu Outbreak | Binary feature indicating outbreak occurrence |

All datasets were merged using the monthly date column.

---

# 🔧 Feature Engineering

Several new features were created to improve prediction performance.

### Bird Flu Outbreak

```
0 = No outbreak

1 = Outbreak
```

### Inflation Rate

Calculated from the Consumer Price Index (CPI).

```
Inflation Rate = Year-over-Year Percentage Change in CPI
```

---

# 📈 Exploratory Data Analysis

The project includes:

- Summary Statistics
- Histograms
- Boxplots
- Line Charts
- Scatter Plots
- Correlation Heatmaps

EDA was performed to understand variable distributions, identify outliers, and examine relationships between predictors and egg prices.

---

# 🤖 Machine Learning Models

The following regression models were trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Tuned Random Forest Regressor

---

# ⚙ Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest model by tuning several parameters, including:

- Number of Trees
- Maximum Depth
- Minimum Samples Split
- Minimum Samples Leaf

---

# 📊 Model Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The best-performing model was selected based on these evaluation metrics.

---

# 🌲 Feature Importance

Feature importance analysis was performed to determine which variables contributed the most to the final model's predictions.

This improves model interpretability by identifying the most influential economic and agricultural factors.

---

# ⏳ Time Series Validation

Because the dataset consists of monthly observations, TimeSeriesSplit validation was performed to evaluate the model using chronological data.

This provides a more realistic estimate of future forecasting performance.

---

# 💻 Interactive Prediction Program

The notebook includes an interactive prediction program that allows users to enter new values for:

- Corn Price
- Soybean Price
- Diesel Price
- CPI
- Inflation Rate
- Birds Affected
- Bird Flu Outbreak

The trained model then predicts the expected retail egg price.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# 📁 Repository Structure

```
egg-price-predictor/
│
├── data/
│   ├── egg_price.csv
│   ├── corn_price.csv
│   ├── soybean_price.csv
│   ├── diesel_price.csv
│   ├── cpi.csv
│   └── bird_flu.csv
│
├── models/
│   ├── egg_price_model.pkl
│   └── preprocessing.pkl
│
├── egg_price_predictor.ipynb
│
└── README.md
```

---

# 🚀 How to Run

1. Clone this repository.

```
git clone https://github.com/bonillasteven/egg-price-predictor.git
```

2. Install the required libraries.

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open

```
egg_price_predictor.ipynb
```

4. Run every notebook cell from top to bottom.

---

# 📌 Future Improvements

Future versions of this project may include:

- XGBoost
- LightGBM
- ARIMA Time-Series Forecasting
- LSTM Neural Networks
- Streamlit Web Application
- Live USDA Data Integration
- SHAP Explainability

---

# 👨‍💻 Author

**Steven A. Bonilla**

Computer Science Student  
Saint Cloud State University

Interested in:

- Artificial Intelligence
- Machine Learning
- Data Science
- Software Engineering

---

## ⭐ If you found this project interesting, consider giving the repository a Star!
