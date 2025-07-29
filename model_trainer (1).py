from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'Support Vector Machine': SVC(
                random_state=42,
                probability=True,
                kernel='rbf'
            )
        }
    
    def train_model(self, model_name, processed_data):
        """
        Train a specific model and return performance metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported")
        
        # Get data
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        
        # Initialize and train model
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return model, metrics
    
    def train_all_models(self, processed_data):
        """
        Train all available models
        """
        results = {}
        trained_models = {}
        
        for model_name in self.models.keys():
            model, metrics = self.train_model(model_name, processed_data)
            results[model_name] = metrics
            trained_models[model_name] = model
        
        return trained_models, results
    
    def get_best_model(self, results):
        """
        Get the best performing model based on accuracy
        """
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        return best_model_name, results[best_model_name]
    
    def compare_models(self, results):
        """
        Create a comparison of all trained models
        """
        comparison = {}
        
        for model_name, metrics in results.items():
            comparison[model_name] = {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'CV Mean': f"{metrics['cv_mean']:.4f}",
                'CV Std': f"{metrics['cv_std']:.4f}"
            }
        
        return comparison
