import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def preprocess_data(self, data):
        """
        Preprocess the Adult dataset for machine learning
        """
        # Make a copy of the data
        df = data.copy()
        
        # Handle missing values (represented as '?' in Adult dataset)
        df = df.replace('?', np.nan)
        
        # Fill missing values with mode for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'income':  # Don't fill target variable
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Remove unnecessary columns if they exist
        columns_to_drop = ['fnlwgt', 'educational-num']  # fnlwgt is a weight, educational-num is redundant with education
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
        
        # Separate features and target
        if 'income' in df.columns:
            X = df.drop('income', axis=1)
            y = df['income']
            
            # Encode target variable
            y = y.map({'<=50K': 0, '>50K': 1})
        else:
            X = df
            y = None
        
        # Encode categorical variables
        categorical_features = X.select_dtypes(include=['object']).columns
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                X[feature] = self.label_encoders[feature].fit_transform(X[feature].astype(str))
            else:
                # For new data, handle unseen categories
                X[feature] = X[feature].astype(str)
                known_categories = set(self.label_encoders[feature].classes_)
                X[feature] = X[feature].apply(
                    lambda x: x if x in known_categories else 'Unknown'
                )
                
                # Add 'Unknown' to encoder if not present
                if 'Unknown' not in known_categories:
                    self.label_encoders[feature].classes_ = np.append(
                        self.label_encoders[feature].classes_, 'Unknown'
                    )
                
                X[feature] = self.label_encoders[feature].transform(X[feature])
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data if target is available
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': self.feature_names,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler
            }
        else:
            # For prediction data
            X_scaled = self.scaler.transform(X)
            return X_scaled
    
    def preprocess_single_input(self, input_data, processed_data_info):
        """
        Preprocess a single input for prediction
        """
        df = input_data.copy()
        
        # Remove columns that were dropped during training
        columns_to_drop = ['fnlwgt', 'educational-num']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
        
        # Encode categorical variables using fitted encoders
        categorical_features = df.select_dtypes(include=['object']).columns
        
        for feature in categorical_features:
            if feature in self.label_encoders:
                df[feature] = df[feature].astype(str)
                
                # Handle unseen categories
                known_categories = set(self.label_encoders[feature].classes_)
                df[feature] = df[feature].apply(
                    lambda x: x if x in known_categories else 'Unknown'
                )
                
                # Add 'Unknown' to encoder if not present
                if 'Unknown' not in known_categories:
                    self.label_encoders[feature].classes_ = np.append(
                        self.label_encoders[feature].classes_, 'Unknown'
                    )
                
                df[feature] = self.label_encoders[feature].transform(df[feature])
        
        # Ensure all features are present and in correct order
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        df = df[self.feature_names]  # Reorder columns
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        return df_scaled
    
    def get_feature_info(self):
        """
        Get information about processed features
        """
        return {
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
