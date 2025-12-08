# Ferocia MLE Take-home: Term Deposit Prediction

An E2E machine learning pipeline that predicts whether a customer will subscribe to a term deposit based on banking data.

## 📊 Project Overview

This project implements a Logistic Regression Model (or Random Forest) for its ML prediction and features the following:
- **Data Preprocessing**: Data analysis, handling missing values, feature engineering, encoding categorical variables
- **Model Training**: Logistic Regression with class balancing for imbalanced data with multiple different dtypes
- **API Serving**: FastAPI application with a customised user-friendly interface
- **Testing**: Unit tests for preprocessing pipeline validation (a few examples)

## 🎯 Key Findings

### Model Performance (Logistic Regression)
| Metric | Score |
|--------|-------|
| Accuracy | ~0.78 |
| Precision | ~0.32 |
| Recall | ~0.71 |
| F1 Score | ~0.44 |
| ROC AUC | ~0.82 |

I wanted to prioritise high Recall over Precision as minimising false negatives is more important in this scenario.

### Data Insights
- **Class Imbalance**: The dataset is heavily imbalanced (~88% "No" vs ~12% "Yes")
- **Missing Values**: `unknown` values in categorical features handled as NaN with imputation
- **Feature Engineering**: Created `first_campaign` indicator from `pdays=-1` flag

### Areas for Future Improvement
1. Try ensemble models (Random Forest, XGBoost, LightGBM)
2. Feature selection to reduce dimensionality
3. Cross-validation for more robust evaluation
4. Deeper analysis into feature relationships 
5. Consider removing `duration` feature (potential data leakage - unknown before call)

---

## 📁 Project Structure

```
Ferocia-MLE-takehome/
├── DATA/
│   ├── dataset.csv              # Raw dataset (semicolon-separated)
│   └── data-dictionary.txt      # Feature descriptions
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Data cleaning & feature engineering
│   │   └── exploring.ipynb      # EDA notebook
│   ├── models/
│   │   └── train.py             # Model training script
│   ├── api/
│   │   ├── app.py               # FastAPI application
│   │   └── index.html           # Web interface
│   └── utils/
│       └── config.py            # Configuration utilities
├── tests/
│   └── test_preprocessing.py    # Unit tests
├── artifacts/                   # Saved model & preprocessor (.pkl)
├── configs/
│   └── config.yaml              # Configuration file
├── AI_TRANSCRIPTS/              # LLM conversation logs
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Ferocia-MLE-takehome.git
cd Ferocia-MLE-takehome
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Running the Pipeline

### Train the Model
```bash
./venv/bin/python -m src.models.train
```

This will:
- Load and preprocess the data
- Train a Logistic Regression model
- Evaluate and print metrics
- Save `model.pkl` and `preprocessor.pkl` to `artifacts/`

### Run Tests
```bash
./venv/bin/python -m pytest tests/test_preprocessing.py -v
```

### Start the API Server
```bash
./venv/bin/python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Then open your browser:
- **Web Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the web interface |
| `/health` | GET | Health check endpoint |
| `/predict` | POST | Make predictions |
| `/docs` | GET | Swagger API documentation |

### Example Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 300,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }'
```

---

## 🤖 AI Tools Used

This project was developed with assistance from **Claude (Anthropic)** via **Cursor IDE**.

### How AI Assisted
- **Project scaffolding**: Setting up initial file structure and boilerplate code
- **Function generation**: Created functions on demand that created high ROI or were out of scope (like transcribing data-dictionary.txt and creating the UI html code
- **Debugging**: Identifying and fixing issues with virtual environment, FastAPI routes, and imports
- **Code review**: Reviewing preprocessing and training pipelines for best practices
- **Documentation**: Generating docstrings and README content templates 

### AI Transcripts
Full conversation logs are available in the `AI_TRANSCRIPTS/` directory for transparency and reproducibility.

---

## 📝 Notes

- The dataset uses **semicolon (`;`) as the delimiter** - this is handled in `load_data()`
- Values of `"unknown"` in the original data are treated as missing values
- The `pdays=-1` indicates the customer was not previously contacted; this is converted to a `first_campaign` feature

---

## 👩‍💻 Author

Audrey Thompson

---

## 📄 License

This project was created as part of a take-home exercise for Ferocia.
