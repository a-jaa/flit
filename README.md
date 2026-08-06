# Credit Risk Assistant

A machine learning application that predicts loan default probability and provides
plain-language explanations to help users understand their credit risk.

Built by Joseph Salvador, Fenet Diriba, Lisha Nishat — Group 14C

## Models
Three models were trained and evaluated:
- Logistic Regression (AUC: 0.715)
- Random Forest (AUC: 0.717)
- LightGBM (AUC: 0.711) ← used in the app

## Dataset
Download from Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club

Use `accepted_2007_to_2018Q4.csv`. The app runs from the saved model file and
does not require the dataset at runtime.

To retrain: run the notebooks in order.

## Run the app
- pip install -r requirements.txt
- streamlit run app.py
