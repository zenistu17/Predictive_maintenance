import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, f1_score, recall_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score, make_scorer, roc_curve
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs("/Users/sujithsuresh/Downloads/Predictive_Maintenance/models", exist_ok=True)
os.makedirs("/Users/sujithsuresh/Downloads/Predictive_Maintenance/plots", exist_ok=True)

# Load processed data
print("Loading data...")
X_train = pd.read_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/X_train.csv")
X_test = pd.read_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/X_test.csv")
y_train = pd.read_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/y_train.csv").values.ravel()
y_test = pd.read_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/y_test.csv").values.ravel()

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Class distribution in training set:", np.bincount(y_train))
print("Class distribution in test set:", np.bincount(y_test))

# Feature importance using multiple methods
def select_features(X_train, y_train, X_test, n_features=20):
    print("\n=== Feature Selection Analysis ===")
    
    # Method 1: XGBoost feature importance
    xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'XGBoost_Importance': xgb_model.feature_importances_
    }).sort_values('XGBoost_Importance', ascending=False)
    
    # Method 2: Random Forest feature importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'RF_Importance': rf_model.feature_importances_
    }).sort_values('RF_Importance', ascending=False)
    
    # Method 3: Recursive Feature Elimination
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
              n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    rfe_support = rfe.support_
    
    # Combine results
    feature_importance = pd.merge(xgb_importance, rf_importance, on='Feature')
    feature_importance['RFE_Selected'] = rfe_support
    feature_importance['Combined_Score'] = (
        feature_importance['XGBoost_Importance'] + 
        feature_importance['RF_Importance'] +
        feature_importance['RFE_Selected'].astype(int)
    )
    
    # Select top features
    top_features = feature_importance.nlargest(n_features, 'Combined_Score')['Feature'].tolist()
    print(f"Selected {len(top_features)} features")
    
    # Visualize top 15 features
    plt.figure(figsize=(12, 8))
    top_15 = feature_importance.nlargest(15, 'Combined_Score')
    sns.barplot(x='Combined_Score', y='Feature', data=top_15)
    plt.title('Top 15 Features by Importance Score')
    plt.tight_layout()
    plt.savefig("/Users/sujithsuresh/Downloads/Predictive_Maintenance/plots/feature_importance.png")
    
    # Return selected features
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    
    return X_train_selected, X_test_selected, top_features

# Create custom evaluation metric for imbalanced data
def custom_f2_score(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    # F2 score weights recall higher than precision
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    return f2

custom_scorer = make_scorer(custom_f2_score)

# Start model development
print("\n=== Starting Enhanced Model Development Pipeline ===")

# 1. Feature selection
X_train_selected, X_test_selected, selected_features = select_features(X_train, y_train, X_test)

# 2. Create time-based cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Apply SMOTE to handle class imbalance
print("\n=== Applying SMOTE for Class Imbalance ===")
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Adjust sampling_strategy as needed
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

print(f"Original class distribution: {Counter(y_train)}")
print(f"Resampled class distribution: {Counter(y_train_resampled)}")

# 3. Train multiple models and maintain a voting ensemble
models = {}
predictions = {}

# 3.1 XGBoost with advanced configuration
print("\n=== Training XGBoost Model ===")
xgb_params = {
    'n_estimators': 500,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'scale_pos_weight': 100,  # Adjusted for class imbalance
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbosity': 0
}

# Fit the model with validation set
xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_test_selected, y_test)],  # Validation set for early stopping
    verbose=True
)

# Save best model
models['xgboost'] = xgb_model
predictions['xgboost'] = xgb_model.predict_proba(X_test_selected)[:, 1]

# 3.2 LightGBM model
print("\n=== Training LightGBM Model ===")
lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'metric': 'auc',
    'early_stopping_rounds': 50,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbosity': -1  # Suppress LightGBM warnings
}

lgb_model = LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_test_selected, y_test)]
)

models['lightgbm'] = lgb_model
predictions['lightgbm'] = lgb_model.predict_proba(X_test_selected)[:, 1]

# 3.3 Random Forest model
print("\n=== Training Random Forest Model ===")
rf_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42
}

rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train_resampled, y_train_resampled)

models['random_forest'] = rf_model
predictions['random_forest'] = rf_model.predict_proba(X_test_selected)[:, 1]

# 4. Create stacked ensemble model
print("\n=== Building Stacked Ensemble Model ===")

# Define base models for stacking (without early stopping)
base_estimators = [
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)),  # Simplified XGBoost
    ('lgb', LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)),  # Simplified LightGBM
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))  # Simplified Random Forest
]

stacked_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
    cv=3
)

# Fit the stacking model
stacked_model.fit(X_train_resampled, y_train_resampled)
models['stacked_ensemble'] = stacked_model
predictions['stacked_ensemble'] = stacked_model.predict_proba(X_test_selected)[:, 1]

# 5. Threshold optimization for each model
print("\n=== Optimizing Decision Thresholds ===")
optimal_thresholds = {}
best_f2_scores = {}

for model_name, y_probs in predictions.items():
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Calculate F2 score for each threshold
    f2_scores = []
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f2 = 5 * precision[i] * recall[i] / (4 * precision[i] + recall[i])
            f2_scores.append(f2)
        else:
            f2_scores.append(0)
    
    # Find optimal threshold
    best_idx = np.argmax(f2_scores)
    if best_idx < len(thresholds):
        optimal_thresholds[model_name] = thresholds[best_idx]
    else:
        optimal_thresholds[model_name] = 0.5  # Default if at the edge
    
    best_f2_scores[model_name] = f2_scores[best_idx]
    
    print(f"{model_name} - Optimal threshold: {optimal_thresholds[model_name]:.3f}, F2 Score: {best_f2_scores[model_name]:.3f}")

# 6. Compare models and select the best one
best_model_name = max(best_f2_scores.items(), key=lambda x: x[1])[0]
best_model = models[best_model_name]
best_threshold = optimal_thresholds[best_model_name]

print(f"\nBest model: {best_model_name} with F2 score: {best_f2_scores[best_model_name]:.3f}")

# 7. Final evaluation of best model
y_probs_best = predictions[best_model_name]
y_pred_best = (y_probs_best >= best_threshold).astype(int)

print("\n=== Final Model Evaluation ===")
print(classification_report(y_test, y_pred_best))
print(f"ROC-AUC: {roc_auc_score(y_test, y_probs_best):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, y_probs_best):.3f}")

# 8. Visualize results
plt.figure(figsize=(12, 10))

# Plot ROC curves
plt.subplot(2, 1, 1)
for model_name, y_probs in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# Plot Precision-Recall curves
plt.subplot(2, 1, 2)
for model_name, y_probs in predictions.items():
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    
    # Mark optimal threshold
    threshold = optimal_thresholds[model_name]
    y_pred = (y_probs >= threshold).astype(int)
    p, r, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    plt.scatter(r, p, marker='o', color='red')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves with Optimal Thresholds')
plt.legend()

plt.tight_layout()
plt.savefig("/Users/sujithsuresh/Downloads/Predictive_Maintenance/plots/model_comparison.png")

# 9. Feature importance of best model
print("\n=== Analyzing Best Model Feature Importance ===")
if best_model_name in ['xgboost', 'lightgbm', 'random_forest']:
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Features from {best_model_name}')
    plt.tight_layout()
    plt.savefig("/Users/sujithsuresh/Downloads/Predictive_Maintenance/plots/best_model_features.png")
    
    print("Top 10 features from best model:")
    print(feature_importance.head(10))

# 10. Save complete pipeline
print("\n=== Saving Final Model and Pipeline ===")

pipeline = {
    'model': best_model,
    'model_name': best_model_name,
    'threshold': best_threshold,
    'selected_features': selected_features,
    'feature_importance': feature_importance.to_dict() if best_model_name in ['xgboost', 'lightgbm', 'random_forest'] else None,
    'metrics': {
        'precision_recall_fscore': precision_recall_fscore_support(y_test, y_pred_best, average='binary'),
        'roc_auc': roc_auc_score(y_test, y_probs_best),
        'pr_auc': average_precision_score(y_test, y_probs_best)
    }
}

# Save all models for ensemble predictions
for model_name, model in models.items():
    joblib.dump(model, f"/Users/sujithsuresh/Downloads/Predictive_Maintenance/models/{model_name}_model.pkl")

# Save the complete pipeline
joblib.dump(pipeline, "/Users/sujithsuresh/Downloads/Predictive_Maintenance/models/advanced_pipeline.pkl")

# Create prediction function
with open("/Users/sujithsuresh/Downloads/Predictive_Maintenance/models/predict_advanced.py", "w") as f:
    f.write("""
import joblib
import pandas as pd
import numpy as np

def predict_failures(data, model_path="/Users/sujithsuresh/Downloads/Predictive_Maintenance/models/advanced_pipeline.pkl", threshold=None):
    '''
    Predict equipment failures from sensor data using the advanced predictive maintenance model
    
    Parameters:
    data : pandas DataFrame - preprocessed sensor data
    model_path : str - path to the saved pipeline
    threshold : float - custom threshold (if None, uses the saved optimal threshold)
    
    Returns:
    dict containing:
        predictions : array - binary predictions (1 = failure predicted)
        probabilities : array - failure probability scores (0-1)
        anomaly_scores : array - anomaly scores for each record
        alert_level : array - categorical risk level ('Low', 'Medium', 'High', 'Critical')
        failure_time_estimate : array - estimated time to failure in hours (if risk is high)
    '''
    # Load pipeline
    pipeline = joblib.load(model_path)
    
    # Extract components
    model = pipeline['model']
    best_threshold = pipeline['threshold']
    selected_features = pipeline['selected_features']
    
    # Use provided threshold or default to best
    threshold = threshold if threshold is not None else best_threshold
    
    # Select features
    if not all(feature in data.columns for feature in selected_features):
        raise ValueError(f"Input data missing some required features: {selected_features}")
    
    X = data[selected_features]
    
    # Get probabilities
    failure_probs = model.predict_proba(X)[:, 1]
    
    # Binary predictions
    predictions = (failure_probs >= threshold).astype(int)
    
    # Calculate anomaly scores (0-1 scale)
    anomaly_scores = 2 * np.abs(failure_probs - 0.5)
    
    # Define risk levels
    risk_levels = np.select(
        [failure_probs < 0.25, failure_probs < 0.5, failure_probs < 0.75, failure_probs >= 0.75],
        ['Low', 'Medium', 'High', 'Critical'],
        default='Unknown'
    )
    
    # Estimate time to failure (simple approach - real implementation would be more sophisticated)
    # This is just a placeholder - would need real degradation modeling
    time_to_failure = np.where(
        failure_probs >= 0.5,
        np.round(100 * (1 - failure_probs)),  # Higher probability means less time to failure
        np.nan
    )
    
    return {
        'predictions': predictions,
        'probabilities': failure_probs,
        'anomaly_scores': anomaly_scores,
        'alert_level': risk_levels,
        'failure_time_estimate': time_to_failure
    }
""")

print("\n=== Model Development Pipeline Complete ===")
print(f"Best model: {best_model_name}")
print(f"Optimal threshold: {best_threshold:.3f}")
print(f"Best F2 score: {best_f2_scores[best_model_name]:.3f}")
print(f"Models saved to /Users/sujithsuresh/Downloads/Predictive_Maintenance/models/")
print(f"Plots saved to /Users/sujithsuresh/Downloads/Predictive_Maintenance/plots/")

# Add code for prediction demonstration
print("\n=== Demonstration: Predicting Failures on Test Data ===")
from sklearn.metrics import confusion_matrix

y_pred = (predictions[best_model_name] >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("/Users/sujithsuresh/Downloads/Predictive_Maintenance/plots/confusion_matrix.png")

# Calculate business metrics
tn, fp, fn, tp = cm.ravel()
print(f"True Positives (Caught Failures): {tp}")
print(f"False Negatives (Missed Failures): {fn}")
print(f"False Positives (False Alarms): {fp}")
print(f"True Negatives (Normal Operation): {tn}")

# Estimate business impact
avg_failure_cost = 10000  # Cost of unplanned downtime
avg_maintenance_cost = 1000  # Cost of planned maintenance
savings = (tp * avg_failure_cost) - (fp * avg_maintenance_cost)
print(f"\nEstimated cost savings: ${savings}")
print(f"Failure catch rate: {tp/(tp+fn)*100:.1f}%")
print(f"False alarm rate: {fp/(fp+tn)*100:.1f}%")