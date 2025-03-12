# Telecom Customer Churn Prediction

A machine learning-based web application that predicts customer churn in the telecommunications industry. The project combines advanced data preprocessing, machine learning modeling, and an interactive web interface to help telecom companies identify customers at risk of churning.

## Features

- **Customer Churn Prediction**: Predicts whether a customer is likely to churn based on various features
- **Value-Based Classification**: Classifies customers into four categories based on value and churn risk:
  - High Value - No Churn
  - High Value - Churn Risk
  - Low Value - No Churn
  - Low Value - Churn Risk
- **Interactive Web Interface**: User-friendly Dash application for real-time predictions
- **Comprehensive Data Processing**: Advanced preprocessing pipeline for handling various data types
- **Model Evaluation**: Detailed evaluation metrics and results analysis

## Project Structure

```
├── Datasets/                    # Data files and resources
├── images/                      # Visualization images
├── model/                       # Trained model files
├── main.py                     # Web application main file
├── preprocessing.py            # Data preprocessing pipeline
├── Model_Grid_Search.py        # Hyperparameter tuning
├── model_evaluate.py           # Model evaluation scripts
├── model_train.py             # Model training scripts
├── CP_project.ipynb           # Development notebook
├── CP_project_complete.ipynb  # Complete analysis notebook
└── evaluation_results.csv     # Model evaluation results
```

## Technology Stack

- **Python Libraries**:
  - Dash: Web application framework
  - Pandas: Data manipulation
  - Scikit-learn: Machine learning
  - SMOTEN: Imbalanced data handling
  - Pickle: Model serialization

## Features Used for Prediction

1. **Customer Information**:
   - Gender
   - Senior Citizen status
   - Partner status
   - Dependents

2. **Services**:
   - Phone Service
   - Multiple Lines
   - Internet Service
   - Online Security
   - Online Backup
   - Device Protection
   - Tech Support
   - Streaming TV/Movies

3. **Contract Details**:
   - Contract Type
   - Paperless Billing
   - Payment Method
   - Monthly Charges
   - Total Charges

## Installation

1. Clone the repository:
   ```bash
   git clone [repository_url]
   cd customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

4. Access the web interface at `http://localhost:8080`

## Usage

1. Access the web interface
2. Enter customer details in the form
3. Click "Predict Churn"
4. View the prediction results and risk assessment

## Model Development

### Pipeline Architecture

<img src="./figures/pipeline.png" width="600" length="400"/>

The pipeline consists of:
- Numerical features: SimpleImputer (mean) → StandardScaler
- Categorical features: SimpleImputer (most frequent) → OrdinalEncoder
- Final stage: GradientBoostingClassifier

### Data Processing Steps

1. **Data Preprocessing**:
   - Missing value imputation
   - Feature encoding
   - Feature scaling
   - Class imbalance handling using SMOTEN

2. **Model Selection**:
   - Grid search for hyperparameter tuning
   - Cross-validation for model evaluation
   - Performance metric analysis

## Key Highlights

### Business Impact
- **Revenue Protection**: The 4-class model approach reduced potential revenue loss by 41% compared to traditional binary classification
- **Customer Segmentation**: Successfully identified high-value customers at risk of churning, enabling targeted retention strategies
- **Cost-Effective**: Prioritizes retention efforts on high-value customers, optimizing resource allocation

### Technical Achievements
- **Novel Approach**: Implemented a 4-class classification system combining customer value and churn probability
- **Model Performance**: Achieved 89% accuracy in predicting high-value customer churn
- **Feature Importance**: Identified key churn indicators:
  1. Contract type (month-to-month vs. long-term)
  2. Tenure length
  3. Monthly charges
  4. Internet service type
  5. Payment method

### Methodology Innovation
- **Value-Based Classification**: Enhanced traditional churn prediction by incorporating customer lifetime value
- **Balanced Optimization**: Model optimized for both prediction accuracy and business impact
- **Interpretable Results**: Clear actionable insights for business stakeholders

## Documentation

- Detailed project report: `st125066_CP_project_report.pdf`
- Project presentation: `st125066_CP_project_Telecom_Churn_Presentation.pdf`
- Development notebooks: `CP_project.ipynb` and `CP_project_complete.ipynb`

## Results

### Model Performance Comparison

| Model | Revenue Loss (False Negatives) |
|-------|--------------------------------|
| XGBoost (2 Class) | $77,152.20 |
| XGBoost (4 Class) | $45,572.10 |
| HGBoost (4 Class) | $49,647.55 |

The 4-class models (separating customers by value) significantly outperform the traditional 2-class approach, reducing potential revenue loss by over 40%. XGBoost with 4-class classification shows the best performance with the lowest revenue loss from false negatives.

Detailed evaluation metrics are available in:
- `evaluation_results.csv`
- `result_summary.csv`

## Future Improvements

1. Real-time model retraining
2. Additional feature engineering
3. Enhanced visualization dashboard
4. API endpoint development
5. Model interpretability tools
6. Batch prediction capabilities
