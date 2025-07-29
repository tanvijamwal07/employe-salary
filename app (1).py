import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizations import create_visualizations
import io

# Page configuration
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

def main():
    st.title("💰 Employee Salary Prediction System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Data Upload & Exploration", "Model Training", "Real-time Prediction", "Model Performance"]
    )
    
    if page == "Data Upload & Exploration":
        data_upload_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Real-time Prediction":
        prediction_page()
    elif page == "Model Performance":
        performance_page()

def data_upload_page():
    st.header("📊 Data Upload & Exploration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload the Adult dataset CSV file"
    )
    
    # Load default data if no file uploaded
    if uploaded_file is None:
        if st.button("Use Default Adult Dataset"):
            try:
                data = pd.read_csv("adult.csv")
                st.session_state.raw_data = data
                st.session_state.data_loaded = True
                st.success("Default dataset loaded successfully!")
                st.rerun()
            except FileNotFoundError:
                st.error("Default dataset not found. Please upload a CSV file.")
                return
    else:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.raw_data = data
            st.session_state.data_loaded = True
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    if st.session_state.data_loaded:
        data = st.session_state.raw_data
        
        # Data overview
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(data))
        with col2:
            st.metric("Total Columns", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", data.duplicated().sum())
        
        # Display first few rows
        st.subheader("Data Preview")
        st.dataframe(data.head(10))
        
        # Data information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes.astype(str),
            'Unique Values': [data[col].nunique() for col in data.columns],
            'Missing Values': [data[col].isnull().sum() for col in data.columns]
        })
        st.dataframe(col_info)
        
        # Data visualizations
        st.subheader("Data Visualizations")
        create_visualizations(data)
        
        # Preprocess data
        if st.button("Preprocess Data"):
            processor = DataProcessor()
            processed_data = processor.preprocess_data(data)
            st.session_state.processed_data = processed_data
            st.success("Data preprocessed successfully!")
            
            st.subheader("Preprocessed Data Info")
            st.write(f"Shape: {processed_data['X_train'].shape}")
            st.write("Features after preprocessing:")
            st.write(list(processed_data['feature_names']))

def model_training_page():
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload and preprocess data first!")
        return
    
    if st.session_state.processed_data is None:
        st.warning("Please preprocess the data first!")
        return
    
    # Model selection
    st.subheader("Select Models to Train")
    models_to_train = st.multiselect(
        "Choose models:",
        ["Random Forest", "Logistic Regression", "Support Vector Machine"],
        default=["Random Forest", "Logistic Regression"]
    )
    
    if st.button("Train Models") and models_to_train:
        trainer = ModelTrainer()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        trained_models = {}
        results = {}
        
        for i, model_name in enumerate(models_to_train):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i + 1) / len(models_to_train))
            
            model, metrics = trainer.train_model(
                model_name, 
                st.session_state.processed_data
            )
            
            trained_models[model_name] = model
            results[model_name] = metrics
        
        st.session_state.trained_models = trained_models
        st.session_state.model_results = results
        st.session_state.models_trained = True
        
        status_text.text("Training completed!")
        st.success("All models trained successfully!")
        
        # Display results
        st.subheader("Training Results")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)
        
        # Best model
        best_model = results_df['accuracy'].idxmax()
        st.success(f"🏆 Best Model: {best_model} (Accuracy: {results_df.loc[best_model, 'accuracy']:.4f})")

def prediction_page():
    st.header("🔮 Real-time Salary Prediction")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
        return
    
    # Model selection for prediction
    selected_model = st.selectbox(
        "Select Model for Prediction:",
        list(st.session_state.trained_models.keys())
    )
    
    st.subheader("Enter Employee Information")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 17, 90, 39)
        workclass = st.selectbox("Work Class", [
            'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
            'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
        ])
        education = st.selectbox("Education", [
            'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
            'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
            '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'
        ])
        marital_status = st.selectbox("Marital Status", [
            'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
            'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
        ])
        occupation = st.selectbox("Occupation", [
            'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
            'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
            'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
            'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
        ])
    
    with col2:
        relationship = st.selectbox("Relationship", [
            'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
        ])
        race = st.selectbox("Race", [
            'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
        ])
        gender = st.selectbox("Gender", ['Female', 'Male'])
        capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
        capital_loss = st.number_input("Capital Loss", 0, 4356, 0)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", [
            'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
            'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
            'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
            'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
            'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
            'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
            'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru',
            'Hong', 'Holand-Netherlands'
        ])
    
    if st.button("Predict Salary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'education': [education],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'gender': [gender],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours_per_week],
            'native-country': [native_country]
        })
        
        # Process input data
        processor = DataProcessor()
        processed_input = processor.preprocess_single_input(
            input_data, 
            st.session_state.processed_data
        )
        
        # Make prediction
        model = st.session_state.trained_models[selected_model]
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction == 1:
            st.success("💰 Predicted Salary: >$50K")
        else:
            st.info("💼 Predicted Salary: ≤$50K")
        
        # Probability
        st.subheader("Prediction Confidence")
        prob_df = pd.DataFrame({
            'Salary Range': ['≤$50K', '>$50K'],
            'Probability': prediction_proba
        })
        
        fig = px.bar(prob_df, x='Salary Range', y='Probability', 
                     title="Prediction Probabilities")
        st.plotly_chart(fig)

def performance_page():
    st.header("📈 Model Performance Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
        return
    
    # Model comparison
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(st.session_state.model_results).T
    
    # Metrics comparison chart
    fig = px.bar(results_df.reset_index(), 
                 x='index', y='accuracy',
                 title="Model Accuracy Comparison",
                 labels={'index': 'Model', 'accuracy': 'Accuracy'})
    st.plotly_chart(fig)
    
    # Detailed metrics table
    st.dataframe(results_df)
    
    # Feature importance (for Random Forest)
    if 'Random Forest' in st.session_state.trained_models:
        st.subheader("Feature Importance (Random Forest)")
        
        rf_model = st.session_state.trained_models['Random Forest']
        feature_names = st.session_state.processed_data['feature_names']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df.head(15), 
                     x='Importance', y='Feature',
                     orientation='h',
                     title="Top 15 Most Important Features")
        st.plotly_chart(fig)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    selected_model_cm = st.selectbox(
        "Select Model for Confusion Matrix:",
        list(st.session_state.trained_models.keys()),
        key="cm_model"
    )
    
    if selected_model_cm in st.session_state.model_results:
        cm = st.session_state.model_results[selected_model_cm]['confusion_matrix']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {selected_model_cm}')
        st.pyplot(fig)
    
    # Download results
    st.subheader("Download Results")
    if st.button("Generate Report"):
        report = generate_report()
        st.download_button(
            label="Download Model Performance Report",
            data=report,
            file_name="model_performance_report.txt",
            mime="text/plain"
        )

def generate_report():
    """Generate a comprehensive model performance report"""
    report = "Employee Salary Prediction - Model Performance Report\n"
    report += "=" * 60 + "\n\n"
    
    results_df = pd.DataFrame(st.session_state.model_results).T
    
    report += "Model Performance Summary:\n"
    report += "-" * 30 + "\n"
    for model, metrics in st.session_state.model_results.items():
        report += f"\n{model}:\n"
        report += f"  Accuracy: {metrics['accuracy']:.4f}\n"
        report += f"  Precision: {metrics['precision']:.4f}\n"
        report += f"  Recall: {metrics['recall']:.4f}\n"
        report += f"  F1-Score: {metrics['f1_score']:.4f}\n"
    
    best_model = results_df['accuracy'].idxmax()
    report += f"\nBest Performing Model: {best_model}\n"
    report += f"Best Accuracy: {results_df.loc[best_model, 'accuracy']:.4f}\n"
    
    return report

if __name__ == "__main__":
    main()
