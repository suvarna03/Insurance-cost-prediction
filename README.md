# Insurance cost prediction
**Problem Statement**
Insurance companies need to accurately predict the cost of health insurance for individuals to set premiums appropriately. However, traditional methods of cost prediction often rely on broad actuarial tables and historical averages, which may not account for the nuanced differences among individuals. By leveraging machine learning techniques, insurers can predict more accurately the insurance costs tailored to individual profiles, leading to more competitive pricing and better risk management.
**EDA and Hypothesis Testing**
*Distribution Analysis:
Plot any and all visualization that could not be made in tableau.
*Correlation Analysis:
Generate a correlation matrix or heatmap to visualize the relationships between all numerical variables.
Focus on the correlation between premium prices and potential predictors to identify strong associations.
*Outlier Detection:
Identify outliers in key variables using IQR (Interquartile Range) method or Z-scores.
Assess the impact of outliers on the overall distribution and consider strategies for handling them.
*Hypothesis Testing:
   T-tests/ANOVA: Use these tests to compare the means of premium prices across different groups defined by categorical variables (e.g., smokers vs. non-smokers, number of surgeries).
   Chi-square tests: Evaluate the association between two categorical variables (e.g., presence of chronic disease and history of cancer in family).
*Regression Analysis: Apply linear regression to test hypotheses about the impact of various predictors on premium prices.
Data Preprocessing:
*Handling Missing Values: Although initial data checks may not show missing values, always prepare to implement strategies for handling them.
*Feature Engineering: Create new features that might improve model performance, such as Body Mass Index (BMI) from height and weight.
Scaling and Encoding: Apply appropriate scaling to numerical features and encoding to categorical features to prepare the data for machine learning algorithms.
**Model Selection:**
Linear Regression: Start with a simple model to establish a baseline for prediction accuracy.
Tree-based Models: Implement models like Decision Trees, Random Forests, and Gradient Boosting Machines for their ability to handle non-linear relationships and feature importance analysis.
Neural Networks: Explore more complex models like neural networks if the initial models show promising results but require more flexibility in capturing interactions.
Model Evaluation and Validation:
Cross-Validation: Use techniques like k-fold cross-validation to ensure that the model performs well across different subsets of the dataset.
Performance Metrics: Depending on the business objective, use metrics like RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), or R² (Coefficient of Determination) to evaluate model performance.
Confidence Intervals/Prediction Intervals: Provide these intervals along with predictions to give users an idea of prediction reliability.
**Final Output**
Random Forest is the best model across both Original & Smoothed targets.
Original Target: Best R² = 0.8609 Smoothed Target: Best R² = 0.7383 MAE is significantly lower for smoothed data, meaning better predictions in absolute terms. Smoothing helps stabilize performance but lowers R² slightly.

The smoothed target reduces variance, making the models less prone to overfitting. However, the R² score slightly drops, indicating it removes some valuable signal. Linear Models (Ridge/Lasso) are not suitable.

R² for Ridge/Lasso is too low (~0.31-0.39), confirming that a linear assumption is insufficient. Decision Tree is prone to overfitting.

The gap between Validation and Test R² is high (0.743 → 0.837 for Original).


 
