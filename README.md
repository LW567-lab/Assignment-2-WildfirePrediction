# Assignment-2-WildfirePrediction

##  Introduction
This project aims to predict forest fire occurrence and severity using machine learning techniques. We use meteorological data such as temperature, humidity, wind speed, and precipitation to build predictive models.

##  Data and Features
- **Dataset**: Forest fire dataset with meteorological variables
- **Features Used**:
  - `Temperature`: Higher temperatures increase fire risk
  - `Humidity`: Lower humidity increases fire spread probability
  - `Wind Speed`: Stronger wind accelerates fire spread
  - `Rain`: Reduces fire risk
  - `Fire Occurrence`: Binary target variable indicating fire presence

##  Models Used
We implemented the following machine learning models:
- **Linear Regression**
- **Random Forest**
- **XGBoost**
- **Stacking Model (Blending Multiple Models)**

##  How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/LW567-lab/ForestFireEDA.git
   cd ForestFireEDA

Assignment 3 – Final Model Refinement, Evaluation, and Interpretability
This section builds on the baseline models developed in Assignment 2 and includes advanced model tuning, validation, and interpretability analysis.

Model Tuning
We used GridSearchCV to optimize hyperparameters for the XGBoost regressor.
Best parameters identified:

learning_rate = 0.01

max_depth = 3

n_estimators = 200

The optimized model achieved the following performance on the test set:

R² Score ≈ 0.61

RMSE ≈ 0.83

Model Evaluation
We applied 5-fold cross-validation to verify model stability and generalization.
Additional diagnostics included:

Residual plot to analyze error distribution

Learning curve to compare training and validation performance

Model Interpretability
SHAP (SHapley Additive Explanations) was used to explain model predictions:

Global feature importance showed fire_occurred, DMC, and RH as top contributors

Local explanation using a waterfall plot demonstrated how each feature affected an individual prediction

Final Notebook and Model
The complete modeling pipeline is documented in assignment3_final.ipynb

The final trained model is saved as random_forest_model.pkl

## Assignment 3 – Final Model Refinement, Evaluation, and Interpretability
This section builds on the baseline models developed in Assignment 2 and includes advanced model tuning, validation, and interpretability analysis.

## Model Tuning
We used `GridSearchCV` to optimize hyperparameters for the XGBoost regressor.  
Best parameters identified:
- `learning_rate = 0.01`
- `max_depth = 3`
- `n_estimators = 200`

The optimized model achieved the following performance on the test set:
- R² Score ≈ 0.61
- RMSE ≈ 0.83

## Model Evaluation
We applied 5-fold cross-validation to verify model stability and generalization.  
Additional diagnostics included:
- Residual plot to analyze error distribution
- Learning curve to compare training and validation performance

## Model Interpretability
`SHAP` (SHapley Additive Explanations) was used to explain model predictions:
- Global feature importance showed `fire_occurred`, `DMC`, and `RH` as top contributors
- Local explanation using a waterfall plot demonstrated how each feature affected an individual prediction

## Final Notebook and Model
- The complete modeling pipeline is documented in `assignment3_final.ipynb`
- The final trained model is saved as `random_forest_model.pkl`

