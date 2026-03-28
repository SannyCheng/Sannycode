# Model Inference Instructions

This folder contains the trained Random Forest model (`random_forest_final.pkl`)
and the label encoders used during training (`label_encoders.pkl`).

## How to run inference

1. Load the trained model using:
   `joblib.load("random_forest_final.pkl")`

2. Prepare input data as a single-row DataFrame.  
   The input must contain all features used during training, including:
   age, gender, sleep_quality_index, brain_fog_level, physical_pain_score,
   stress_level, depression_phq9_score, fatigue_severity_scale_score,
   pem_duration_hours, hours_of_sleep_per_night, pem_present,
   work_status, social_activity_level, exercise_frequency,
   meditation_or_mindfulness.

3. Load the saved label encoders (`label_encoders.pkl`)  
   and apply them to encode categorical features before prediction.

4. Call `model.predict()` on the processed input to obtain the output class  
   (0 = Depression, 1 = ME/CFS, 2 = Both).

This model is intended for demonstration only and should be used with clinician oversight.

