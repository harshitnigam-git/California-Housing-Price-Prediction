# California Housing Price Prediction

This project predicts housing prices in California using a Random Forest Regressor with a full preprocessing pipeline.

## Features
- Preprocessing pipeline for numerical and categorical features
- Handles missing values and scales numerical attributes
- One-hot encoding for categorical data
- Stratified train-test split based on median income categories

## Model
- Random Forest Regressor
- Trained on 80% of the dataset
- Evaluated on 20% test data

## Performance
- Training Accuracy (R²): 97.49%
- Test Accuracy (R²): 82.91%

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python housing_prediction.py`
4. Outputs predictions in `output.csv`

## Tools Used
- Python
- Pandas
- Scikit-learn
- Random Forest Regressor

## Author
Harshit Nigam