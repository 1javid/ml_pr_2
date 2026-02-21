# Airline Passenger Satisfaction Analysis

End-to-end ML study on airline passenger satisfaction data covering binary & multinomial logistic regression, LDA, QDA, Naïve Bayes, and regression models.

## Project Structure

```
ML_PR_2/
├── projecttwin.ipynb   # Main analysis notebook (EDA → models → evaluation)
├── app.py              # Interactive Streamlit UI (separate module)
├── requirements.txt    # Python dependencies for the UI
├── report.tex          # LaTeX source for the written report
└── A2_ML_REPORT.pdf    # Compiled PDF report
```

## Running the Interactive UI

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the app**

   ```bash
   streamlit run app.py
   ```

   The app opens at `http://localhost:8501` and includes four pages:

   | Page | Description |
   |---|---|
   | 🏠 Overview | Dataset summary, pipeline steps, key findings |
   | 🔮 Live Predictor | Enter passenger details and get a real-time satisfaction prediction |
   | 📊 Model Comparison | Accuracy / ROC-AUC / F1 charts, ROC curves, classification reports |
   | 📋 Example Results | Three worked test-set examples with confusion matrix |

## Running the Notebook

Open `projecttwin.ipynb` in Jupyter or VS Code. The notebook expects the Kaggle *Airline Passenger Satisfaction* dataset (train/test CSV files) to be available in the working directory.

```bash
jupyter notebook projecttwin.ipynb
```

## Models Covered

- **Binary Classification** — Logistic Regression · LDA · QDA · Gaussian Naïve Bayes  
- **Multinomial Classification** — Logistic Regression (Business / Eco / Eco Plus)  
- **Regression** — OLS Linear Regression · Poisson GLM (target: flight distance)

## Best Result

LDA with optimal threshold achieved **87% accuracy** and **ROC-AUC 0.944** on the held-out test set (n = 25,976).
