# Predictive Maintenance Project

![Predictive Maintenance](https://img.shields.io/badge/Predictive-Maintenance-blue)  
![Python](https://img.shields.io/badge/Python-3.11-green)  
![License](https://img.shields.io/badge/License-MIT-orange)

---

## **Overview**

This project focuses on building a predictive maintenance system using machine learning. The goal is to predict equipment failures based on sensor data, enabling proactive maintenance and reducing downtime. The project includes:

- **Data Generation**: Synthetic sensor data with realistic failure patterns.
- **Data Preprocessing**: Feature engineering, resampling, and normalization.
- **Model Training**: Training and evaluation of multiple machine learning models (XGBoost, LightGBM, Random Forest, and Stacked Ensemble).
- **Model Deployment**: Saving the best model and creating a prediction function for real-time use.

---

## **Project Structure**
```
Predictive_Maintenance/
├── data/
│   ├── raw_data.csv                # Raw synthetic sensor data
│   ├── X_train.csv                 # Preprocessed training features
│   ├── X_test.csv                  # Preprocessed test features
│   ├── y_train.csv                 # Training labels
│   ├── y_test.csv                  # Test labels
│   └── feature_names.txt           # List of selected features
├── models/
│   ├── xgboost_model.pkl           # Trained XGBoost model
│   ├── lightgbm_model.pkl          # Trained LightGBM model
│   ├── random_forest_model.pkl     # Trained Random Forest model
│   ├── stacked_ensemble_model.pkl  # Trained Stacked Ensemble model
│   ├── advanced_pipeline.pkl       # Complete pipeline (best model + metadata)
│   └── predict_advanced.py         # Prediction function for deployment
├── plots/
│   ├── feature_importance.png      # Top 15 features by importance
│   ├── model_comparison.png        # ROC and Precision-Recall curves
│   ├── best_model_features.png     # Top 15 features from the best model
│   └── confusion_matrix.png        # Confusion matrix for the best model
├── scripts/
│   ├── data_generation.py          # Script to generate synthetic data
│   ├── data_preprocessing.py       # Script to preprocess data
│   └── model_training.py           # Script to train and evaluate models
│   └── combine_data.py             # Script to combine all the data for visualization
└── README.md                       # Project documentation
```

---

## **Installation**

1. Clone the repository:
   
```bash
git clone https://github.com/zenistu17/Predictive_Maintenance.git
cd Predictive_Maintenance
```

2. Create a virtual environment:

```bash
python -m venv predictive_maintenance_env
source predictive_maintenance_env/bin/activate  # On Windows: predictive_maintenance_env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## **Usage**

### 1. Data Generation
Generate synthetic sensor data with realistic failure patterns:

```bash
python scripts/data_generation.py
```

### 2. Data Preprocessing
Preprocess the data and split it into training and test sets:

```bash
python scripts/data_preprocessing.py
```

### 3. Model Training
Train and evaluate machine learning models:

```bash
python scripts/model_training.py
```

### 4. Model Deployment
Use the saved pipeline to make predictions on new data:

```python
from models.predict_advanced import predict_failures

# Load new data
new_data = pd.read_csv("path/to/new_data.csv")

# Make predictions
predictions = predict_failures(new_data)
print(predictions)
```

## **Results**

### Model Performance
**Best Model**: XGBoost
- F2 Score: 1.000
- ROC-AUC: 1.000
- PR-AUC: 1.000

### Business Impact
- Failure Catch Rate: 100.0%
- False Alarm Rate: 0.0%
- Estimated Cost Savings: $10,000

### Feature Importance
The top 5 features for failure prediction are:
1. time_to_next_failure
2. vibration_rolling_std_5
3. time_since_last_failure
4. pressure_deviation
5. temperature_rolling_mean_5

## **Contributing**

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.

## **Contact**

- Name: Sujith S
- GitHub: zenistu17
- LinkedIn: Sujith S
- Email: sujithsures@gmail.com

## **Acknowledgments**

- Thanks to the open-source community for providing the tools and libraries used in this project.
