# Churn Risk Intelligence
*Customer Churn Prediction & Retention Analytics*

A comprehensive machine learning solution for predicting customer churn in telecommunications, enabling businesses to proactively identify at-risk customers and implement targeted retention strategies. This project demonstrates end-to-end pipeline development from data exploration to production-ready models with actionable business insights.

## Business Impact

**Problem**: Customer acquisition costs are 5-25x higher than retention costs, yet most companies struggle to identify which customers are likely to churn before it's too late.

**Solution**: Predictive analytics system that identifies high-risk customers with 82% accuracy, enabling proactive retention campaigns and optimized marketing spend.

**Value Delivered**:
- Identifies high-risk customers for targeted retention campaigns
- Reduces revenue loss from customer attrition
- Optimizes marketing budget allocation with precision targeting
- Provides actionable insights for customer retention strategies

## Key Results

| Model | Accuracy | Precision (Churn) | Recall (Churn) | Business Use Case |
|-------|----------|------------------|----------------|-------------------|
| **Logistic Regression** | **82.19%** | **69%** | **60%** | **Balanced Performance** |
| KNN (k=15) | 77.93% | 60% | 51% | Standard Classification |
| Logistic + SMOTE | 76.86% | 55% | 72% | Higher Recall Priority |
| KNN + SMOTE | 72.89% | 49% | **81%** | **Maximum Churn Detection** |

**Key Findings**:
- **Best Overall Performance**: Logistic Regression achieves highest accuracy and precision
- **Best for Churn Detection**: KNN + SMOTE catches 81% of actual churners
- **Business Trade-off**: Choice between precision (fewer false alarms) vs recall (catching more churners)

## Dataset Information

**Source**: Telco Customer Churn Dataset
- **Size**: 7,043 customers × 21 features  
- **Target**: Customer churn (Yes/No)
- **Class Distribution**: 26.5% churn rate (imbalanced dataset)

**Feature Categories**:
- **Demographics**: Gender, age range, partner/dependent status
- **Account Information**: Contract type, payment method, tenure, billing preferences
- **Services**: Phone, internet, security, backup, streaming services
- **Financial**: Monthly charges, total charges

## Technical Architecture

### Methodology Overview
1. **Exploratory Data Analysis**: Target distribution analysis and feature correlation mapping
2. **Data Preprocessing**: Missing value imputation, categorical encoding, feature scaling
3. **Feature Engineering**: One-hot encoding, data leakage detection, train-test splitting
4. **Model Development**: Logistic Regression and K-Nearest Neighbors with hyperparameter optimization
5. **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
6. **Model Evaluation**: Comprehensive metrics analysis with business impact assessment

### Key Technical Features
- **Modular Architecture**: Separation of concerns with dedicated modules for each pipeline stage
- **Data Leakage Detection**: Automated identification of features with perfect correlation to target
- **Class Imbalance Handling**: SMOTE implementation for balanced model training
- **Hyperparameter Optimization**: GridSearchCV for optimal model parameters
- **Comprehensive Evaluation**: Multiple metrics with business-focused interpretation

## Technologies Used

- **Python 3.8+**: Core programming language
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib
- **Development**: Jupyter notebooks for exploration

## Project Structure

```
churn-risk-intelligence/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── src/                              # Source code modules
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration and constants
│   ├── data_processing.py            # Data loading, cleaning, and EDA
│   ├── feature_engineering.py       # Feature transformations and preprocessing
│   ├── model_training.py             # Model definitions and training pipeline
│   ├── evaluation.py                 # Model evaluation and results analysis
│   └── main.py                       # Main pipeline orchestrator
├── data/
│   └── Customer-Churn.csv           # Dataset (download required)
├── models/                          # Saved model files (.pkl)
├── results/                         # Output files, plots, and reports
└── notebooks/                       # Exploratory analysis (optional)
```

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/abzanganeh/churn_risk_intelligence.git
cd churn-risk-intelligence
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download dataset**:
   - Download the Telco Customer Churn dataset
   - Place `Customer-Churn.csv` in the `data/` directory

### Usage

**Run complete pipeline**:
```bash
python src/main.py
```

**Run individual components**:
```python
from src.data_processing import main_data_processing
from src.feature_engineering import main_feature_engineering
from src.model_training import main_model_training
from src.evaluation import main_evaluation

# Step-by-step execution
cleaned_df, encoded_df = main_data_processing()
processed_data = main_feature_engineering(cleaned_df)
training_results = main_model_training(processed_data)
evaluator = main_evaluation(training_results, processed_data['X_test'], processed_data['y_test'])
```

## Results and Outputs

After running the pipeline, the following files are generated:

### Models (`models/` directory)
- `logistic_regular_model.pkl` - Logistic Regression (standard training)
- `knn_regular_model.pkl` - K-Nearest Neighbors (standard training)  
- `logistic_smote_model.pkl` - Logistic Regression (SMOTE-enhanced)
- `knn_smote_model.pkl` - KNN (SMOTE-enhanced)
- `knn_optimized_model.pkl` - Hyperparameter-optimized KNN

### Analysis Reports (`results/` directory)
- `model_comparison.csv` - Performance metrics comparison table
- `confusion_matrices.png` - Visual confusion matrix analysis
- `churn_distribution.png` - Target variable distribution plot
- `evaluation_summary.txt` - Comprehensive evaluation summary
- Individual model detailed reports with business interpretations

## Business Recommendations

### For High-Value Customer Businesses
- **Use KNN + SMOTE model** for maximum churn detection (81% recall)
- Implement comprehensive retention campaigns for all flagged customers
- Focus on customers with fiber optic internet and month-to-month contracts

### For Cost-Conscious Operations  
- **Use Logistic Regression model** for efficient targeting (69% precision)
- Prioritize customers with electronic check payments and high monthly charges
- Develop automated retention workflows for scalability

### Key Risk Factors to Monitor
1. **Contract Type**: Month-to-month contracts show highest churn rates
2. **Internet Service**: Fiber optic users demonstrate elevated churn risk
3. **Payment Method**: Electronic check payments correlate with increased churn
4. **Tenure**: New customers (< 6 months) require attention
5. **Service Bundle**: Customers without online security/backup services at higher risk

## Model Performance Details

### Classification Metrics
- **Accuracy**: Overall prediction correctness (80-82% range)
- **Precision**: Efficiency of churn predictions (49-69% range)
- **Recall**: Ability to catch actual churners (56-81% range)
- **F1-Score**: Balanced precision-recall metric (0.57-0.61 range)

### Business Metrics Interpretation
- **Churn Detection Rate**: Percentage of actual churners correctly identified
- **Marketing Efficiency**: Accuracy when predicting churn (inverse of false alarm rate)
- **Campaign ROI**: Cost-benefit analysis based on precision-recall trade-offs

## Future Enhancements

- [ ] **Advanced Models**: Random Forest, XGBoost, Neural Networks
- [ ] **Feature Engineering**: Interaction terms, polynomial features, customer segmentation
- [ ] **Time Series Analysis**: Seasonal churn patterns and temporal features
- [ ] **Real-time Scoring**: API deployment for live predictions
- [ ] **A/B Testing Framework**: Retention campaign effectiveness measurement
- [ ] **Ensemble Methods**: Voting classifier for improved performance
- [ ] **Customer Lifetime Value**: Integration with CLV models for prioritization

## Technical Implementation Notes

### Data Quality Considerations
- Missing values in `TotalCharges` (11 records) handled via mean imputation
- Categorical variables one-hot encoded with `drop_first=True` to avoid multicollinearity
- Data leakage detection implemented to identify perfectly correlated features

### Model Training Details
- Train-test split: 80-20 with stratification to maintain class distribution
- Feature scaling applied to numerical variables (`tenure`, `MonthlyCharges`, `TotalCharges`)
- Hyperparameter optimization via 5-fold cross-validation
- SMOTE applied only to training data to prevent data leakage

### Performance Validation
- Cross-validation scores reported for model reliability assessment
- Confusion matrices analyzed for false positive/negative business impact
- Multiple evaluation metrics provided for comprehensive model assessment

## Contributing

This project demonstrates applied science methodology for business problem-solving. Contributions that enhance the predictive accuracy, business applicability, or technical implementation are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Project Status**: Complete | **Last Updated**: September 2025

*This project showcases end-to-end machine learning pipeline development with focus on business impact and actionable insights for customer retention strategies.*