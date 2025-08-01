# Employee Salary Prediction System

A comprehensive Streamlit-based machine learning application for predicting employee salaries using the Adult dataset (Census Income dataset). This application provides end-to-end functionality from data exploration to real-time predictions.

## Features

### 🔍 Data Exploration
- Interactive data upload and visualization
- Comprehensive statistical analysis
- Income distribution analysis
- Feature correlation heatmaps
- Demographic insights

### 🤖 Machine Learning Models
- **Random Forest Classifier** - Ensemble learning for robust predictions
- **Logistic Regression** - Linear classification baseline
- **Support Vector Machine** - Complex decision boundary handling

### 📊 Interactive Visualizations
- Age demographics by income level
- Education vs income analysis
- Work hours distribution
- Gender-based income patterns
- Feature importance analysis

### 🔮 Real-time Predictions
- User-friendly input interface
- Instant salary predictions
- Confidence probability scores
- Interactive prediction results

### 📈 Performance Analytics
- Model comparison metrics
- Confusion matrix analysis
- Feature importance rankings
- Downloadable performance reports

## Dataset

The application uses the Adult dataset (also known as Census Income dataset) which contains demographic and employment information to predict whether an individual's income exceeds $50K annually.

**Features include:**
- Age, education, occupation
- Marital status, relationship
- Race, gender, native country
- Work hours, capital gains/losses

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/employee-salary-prediction.git
cd employee-salary-prediction
```

2. Install dependencies:
```bash
pip install streamlit pandas numpy plotly scikit-learn seaborn matplotlib
```

3. Run the application:
```bash
streamlit run app.py
```

### Replit Deployment

1. Import this repository to Replit
2. The application will automatically install dependencies
3. Click "Run" to start the Streamlit server
4. Access your app at the provided URL

## Usage

### 1. Data Upload & Exploration
- Navigate to "Data Upload & Exploration"
- Upload your CSV file or use the default Adult dataset
- Explore interactive visualizations and statistics
- Click "Preprocess Data" to prepare for training

### 2. Model Training
- Go to "Model Training" section
- Select which models to train
- Monitor training progress
- Review performance metrics

### 3. Make Predictions
- Visit "Real-time Prediction" page
- Enter employee information using the form
- Get instant salary predictions with confidence scores

### 4. Analyze Performance
- Check "Model Performance" for detailed analysis
- Compare model accuracy and metrics
- View feature importance rankings
- Download comprehensive reports

## Architecture

### Core Components

- **`app.py`** - Main Streamlit application and UI orchestration
- **`data_processor.py`** - Data preprocessing and feature engineering
- **`model_trainer.py`** - Machine learning model training and evaluation
- **`visualizations.py`** - Interactive data visualization functions

### Data Processing Pipeline

1. **Data Validation** - Verify structure and required columns
2. **Missing Value Handling** - Replace '?' with appropriate values
3. **Feature Encoding** - Label encoding for categorical variables
4. **Feature Scaling** - StandardScaler for numerical features
5. **Train-Test Split** - Balanced data splitting for validation

### Model Training Workflow

1. **Preprocessing** - Clean and transform raw data
2. **Multiple Algorithm Training** - Parallel model training
3. **Performance Evaluation** - Comprehensive metrics calculation
4. **Model Comparison** - Side-by-side performance analysis

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualizations**: Plotly, Seaborn, Matplotlib
- **Deployment**: Replit, GitHub

## Model Performance

The application typically achieves:
- **Random Forest**: ~85% accuracy
- **Logistic Regression**: ~80% accuracy  
- **SVM**: ~82% accuracy

*Performance may vary based on data quality and preprocessing steps.*

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository for the Adult dataset
- Streamlit team for the excellent web framework
- Scikit-learn contributors for machine learning tools

