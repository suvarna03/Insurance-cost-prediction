import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Read the dataset
df = pd.read_csv(r"C:\Users\RAHUL MANGUTKAR\OneDrive\Desktop\GitLearning\project\Insurance-cost-prediction\data\insurance.csv")

## Create IBM feature: BMI based on Height and Weight
df['IBM'] = np.round(df['Weight'] / (df['Height'] / 100) ** 2)

# Apply Log Transformation on IBM feature
df['IBM'] = np.log(df['IBM'])

# Drop the 'Height' and 'Weight' columns after creating IBM
df.drop(columns=['Height', 'Weight'], inplace=True)

# Label Encoding for categorical columns
label_encoder = LabelEncoder()

# Encode the binary categorical features (Yes/No)
df['Diabetes'] = label_encoder.fit_transform(df['Diabetes'])
df['BloodPressureProblems'] = label_encoder.fit_transform(df['BloodPressureProblems'])
df['AnyTransplants'] = label_encoder.fit_transform(df['AnyTransplants'])
df['AnyChronicDiseases'] = label_encoder.fit_transform(df['AnyChronicDiseases'])
df['KnownAllergies'] = label_encoder.fit_transform(df['KnownAllergies'])
df['HistoryOfCancerInFamily'] = label_encoder.fit_transform(df['HistoryOfCancerInFamily'])

# Label Encoding for 'NumberOfMajorSurgeries'
df['NumberOfMajorSurgeries'] = label_encoder.fit_transform(df['NumberOfMajorSurgeries'])

# Define the feature columns and target
X = df.drop(columns=['PremiumPrice'])  # Feature columns
y_original = df['PremiumPrice']  # Target column

# Train-Validation-Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y_original, test_size=0.3, random_state=42)

# Split the temp set into validation and test (33% of the temp set will be test, remaining will be validation)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Apply Standardization
scaler = StandardScaler()

# Fit and transform on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test data using the same scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning for Random Forest
param_grid = {'max_depth': [3, 5, 10, 15, 20]}

rf = RandomForestRegressor(n_estimators=100)
rf_grid = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train_scaled, y_train)

# Get the best Random Forest model
best_rf = rf_grid.best_estimator_

# Train the best Random Forest model
best_rf.fit(X_train_scaled, y_train)

# Save the best Random Forest model and scaler
joblib.dump((best_rf, scaler), "random_forest_model.pkl")
print("Random Forest model and scaler saved successfully!")
