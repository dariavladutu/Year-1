
# NAC Breda Player Recruitment Project  
### Year 1 – Block B | Applied Data Science & AI | Breda University of Applied Sciences  
**Author:** Daria-Elena Vlăduțu  

---

## Project Overview  
This project tackled a real-world challenge from NAC Breda Football Club: building a machine learning system to support scouting efforts by identifying affordable and high-performing defenders under a strict transfer budget.

The goal was to help NAC optimize recruitment by evaluating performance metrics and predicting **successful defensive actions per 90 minutes**, enabling smarter decision-making with data.

---

## Key Objectives  
- Combine football performance and transfer market data to evaluate players.  
- Predict on-pitch performance with a high degree of accuracy.  
- Recommend players under a €300,000 transfer cap.  

---

## Workflow Summary

### 1. **Data Management & Understanding**
- Combined multiple Excel sources from a nested directory structure into a unified dataframe.
- Cleaned null values and standardized inconsistent data.
- Encoded categorical variables and normalized continuous ones.

### 2. **Exploratory Data Analysis (EDA)**
- Identified skewed distributions, class imbalances, and variable correlations.
- Highlighted top players, teams, and key performance drivers visually.
- Visualized salary distribution, playing time, and positional breakdowns.

### 3. **Feature Selection**
- Used `RandomForestRegressor` feature importance scores to select key predictors.
- Manually curated features for model performance, reducing overfitting risk.

### 4. **Modeling**
- Trained and compared multiple regressors:
  - **Gradient Boosting Regressor** (final model)
  - Random Forest
  - Linear Regression
- Hyperparameter tuning with `GridSearchCV`.

### 5. **Evaluation**
- R²: **0.99**  
- RMSE: **0.193**  
- MSE: **0.037**  
- Residual plots and scatter charts confirmed accurate predictive power.

---

## Skills Demonstrated
- Data engineering with `os.walk`, `pandas`, and `Excel`
- Preprocessing: imputation, encoding, normalization
- Supervised ML (regression), feature selection, hyperparameter tuning
- EDA with `seaborn`, `matplotlib`, and correlation matrices
- Use of `scikit-learn` pipelines and metrics
- Documentation with APA-style reporting and structured project tracking

---

## Tools Used
- **Python**: `pandas`, `NumPy`, `scikit-learn`, `matplotlib`, `seaborn`  
- **Jupyter Notebooks**  
- **Excel** (for raw data inspection and early cleaning)  
- **Git** & GitHub for version control and collaboration  

---

## Repository Structure

```
nac-breda-ml-recruitment/
├── data/                      # Cleaned and raw datasets
├── notebooks/                
│   └── Final_Deliverable_Y1BlockB.ipynb  # All modeling and analysis
├── reports/
│   └── Vladutu_Daria_236578_Year1BlockBReport.pdf
├── logs/
│   ├── Learning Log 2023-24Y1B ADS_AI.pptx
│   └── Work Log 2023-24Y1B ADS_AI.xlsx
└── README.md
```

---

## Takeaway  
By the end of Block B, I gained confidence in applying the full CRISP-DM cycle from business understanding through to modeling and reporting. This project emphasized not just technical skill, but the importance of model explainability and stakeholder impact—skills I now carry into every new AI project.
