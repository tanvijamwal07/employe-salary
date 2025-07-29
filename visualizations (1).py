import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def create_visualizations(data):
    """
    Create comprehensive visualizations for the Adult dataset
    """
    
    # Income distribution
    st.subheader("Income Distribution")
    if 'income' in data.columns:
        income_counts = data['income'].value_counts()
        fig = px.pie(values=income_counts.values, names=income_counts.index,
                     title="Income Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution by income
    st.subheader("Age Distribution by Income")
    if 'age' in data.columns and 'income' in data.columns:
        fig = px.histogram(data, x='age', color='income', 
                          title="Age Distribution by Income Level",
                          nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    
    # Education vs Income
    st.subheader("Education Level vs Income")
    if 'education' in data.columns and 'income' in data.columns:
        education_income = pd.crosstab(data['education'], data['income'], normalize='index') * 100
        fig = px.bar(education_income, 
                     title="Income Distribution by Education Level (%)",
                     labels={'value': 'Percentage', 'index': 'Education Level'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Work hours vs Income
    st.subheader("Work Hours vs Income")
    if 'hours-per-week' in data.columns and 'income' in data.columns:
        fig = px.box(data, x='income', y='hours-per-week',
                     title="Work Hours Distribution by Income Level")
        st.plotly_chart(fig, use_container_width=True)
    
    # Gender and Income
    st.subheader("Gender vs Income")
    if 'gender' in data.columns and 'income' in data.columns:
        gender_income = pd.crosstab(data['gender'], data['income'], normalize='index') * 100
        fig = px.bar(gender_income,
                     title="Income Distribution by Gender (%)",
                     labels={'value': 'Percentage', 'index': 'Gender'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Marital Status vs Income
    st.subheader("Marital Status vs Income")
    if 'marital-status' in data.columns and 'income' in data.columns:
        marital_income = pd.crosstab(data['marital-status'], data['income'])
        fig = px.bar(marital_income,
                     title="Income Distribution by Marital Status",
                     labels={'value': 'Count', 'index': 'Marital Status'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Occupation analysis
    st.subheader("Top Occupations")
    if 'occupation' in data.columns:
        # Remove missing values for cleaner visualization
        occupation_data = data[data['occupation'] != '?']['occupation'] if '?' in data['occupation'].values else data['occupation']
        occupation_counts = occupation_data.value_counts().head(10)
        fig = px.bar(x=occupation_counts.values, y=occupation_counts.index,
                     orientation='h',
                     title="Top 10 Occupations",
                     labels={'x': 'Count', 'y': 'Occupation'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap for numerical features
    st.subheader("Numerical Features Correlation")
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 1:
        corr_matrix = data[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Correlation Matrix of Numerical Features")
        st.pyplot(fig)
    
    # Capital gain/loss analysis
    if 'capital-gain' in data.columns and 'capital-loss' in data.columns:
        st.subheader("Capital Gain vs Capital Loss")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Capital gain distribution
            capital_gain_non_zero = data[data['capital-gain'] > 0]['capital-gain']
            if len(capital_gain_non_zero) > 0:
                fig = px.histogram(capital_gain_non_zero, 
                                 title="Capital Gain Distribution (Non-zero values)",
                                 nbins=20)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Capital loss distribution
            capital_loss_non_zero = data[data['capital-loss'] > 0]['capital-loss']
            if len(capital_loss_non_zero) > 0:
                fig = px.histogram(capital_loss_non_zero,
                                 title="Capital Loss Distribution (Non-zero values)",
                                 nbins=20)
                st.plotly_chart(fig, use_container_width=True)
    
    # Country analysis
    st.subheader("Native Country Analysis")
    if 'native-country' in data.columns:
        # Top countries
        country_data = data[data['native-country'] != '?']['native-country'] if '?' in data['native-country'].values else data['native-country']
        country_counts = country_data.value_counts().head(10)
        
        fig = px.bar(x=country_counts.values, y=country_counts.index,
                     orientation='h',
                     title="Top 10 Native Countries",
                     labels={'x': 'Count', 'y': 'Country'})
        st.plotly_chart(fig, use_container_width=True)

def create_model_comparison_chart(results):
    """
    Create a comparison chart for model performance
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    models = list(results.keys())
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [results[model][metric] for model in models]
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=models,
            y=values
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group'
    )
    
    return fig

def create_feature_importance_chart(model, feature_names, top_n=15):
    """
    Create feature importance chart
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig = px.bar(importance_df, 
                     x='Importance', y='Feature',
                     orientation='h',
                     title=f"Top {top_n} Most Important Features")
        return fig
    else:
        return None
