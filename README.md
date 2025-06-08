
# Titanic ML Assignment - Supervised Learning Flow

This project presents a full supervised machine learning pipeline applied to the classic Titanic dataset.  
The goal is to predict passenger survival using models trained on historical data.

## 📁 Files

- `Assignment2_supervised_learning_flow.ipynb`: The main Jupyter notebook containing the full ML flow.
- `titanic_train.csv`: Training dataset.
- `titanic_test.csv`: Testing dataset.
- `titanic_submission.csv`: Final predictions based on the best-performing model.

## 🧠 Supervised Learning Flow

### Part 1 - Preparation
- Load and inspect Titanic datasets
- Handle missing values (e.g. fill `Age` with mean, `Embarked` with mode)
- Convert categorical values (e.g. `Sex`, `Embarked`) to numeric

### Part 2 - Feature Engineering
- Create new features:
  - `FamilySize = SibSp + Parch + 1`
  - `FarePerPerson = Fare / FamilySize`

### Part 3 - Experiments
- Models used: Logistic Regression, Random Forest, Gradient Boosting
- Evaluate with cross-validation (`StratifiedKFold`)
- Use `GridSearchCV` to find best hyperparameters (based on `F1 macro`)

### Part 4 - Training
- Retrain best model (Random Forest) on full training set using optimal hyperparameters

### Part 5 - Prediction & Evaluation
- Predict survival on test set using the trained model
- Save predictions in `titanic_submission.csv`

## ✅ How to Run

1. Open the notebook in Jupyter
2. Run all cells in order
3. Ensure `titanic_train.csv` and `titanic_test.csv` are in the same directory
4. Final predictions will be saved to `titanic_submission.csv`

## 🔧 Requirements

This project uses the following Python libraries:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

Install with:
```bash
pip install -r requirements.txt
