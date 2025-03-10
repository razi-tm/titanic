# Titanic Survival Prediction using Random Forest

## Overview
This project applies machine learning to predict passenger survival on the Titanic. The dataset is processed, and a Random Forest classifier is trained to predict whether a passenger survived based on available features.

## Dataset
The dataset used is the Titanic dataset from Kaggle, which contains passenger details such as age, gender, fare, and class.

## Steps
1. **Read the CSV file**: The dataset is loaded using `pandas.read_csv()`.
2. **Preprocess the data**:
   - The target variable (`Survived`) is extracted.
   - Unnecessary columns (`Name`, `Ticket`, `Cabin`) are dropped.
   - Categorical features are converted into numerical representations using one-hot encoding (`pd.get_dummies()`).
   - Missing values in the `Age` column are filled with the mean age.
3. **Split the dataset**:
   - The dataset is split into training (70%) and testing (30%) sets using `train_test_split()`.
4. **Train the model**:
   - A `RandomForestClassifier` with 100 trees and a max depth of 5 is trained on the training data.
5. **Make Predictions**:
   - The trained model predicts survival on the test set.
6. **Evaluate Accuracy**:
   - The model's accuracy is calculated using `np.mean(predictions == y_test)`, achieving ~85% accuracy.
7. **Save Results**:
   - Predictions can be saved to a CSV file for submission.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy scikit-learn
```

## Future Improvements
- Tune hyperparameters for better performance.
- Use feature engineering to extract more insights from existing data.
- Try different models like Logistic Regression, XGBoost, or Neural Networks.

