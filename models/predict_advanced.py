
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
