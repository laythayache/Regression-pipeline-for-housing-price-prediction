"""
Model Selection and Training Module
Handles model selection, hyperparameter tuning, training, and cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        self.best_params = None
        self.cv_results = {}
        self.training_history = []
        
    def load_engineered_data(self, filepath):
        """Load engineered data"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Engineered data loaded. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading engineered data: {str(e)}")
            raise
    
    def prepare_data(self, df, target_column='price', test_size=0.2, val_size=0.1):
        """Prepare data for training with train/validation/test splits"""
        logger.info("Preparing data for training...")
        
        # Separate features and target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number])
        
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: separate training and validation sets
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        logger.info(f"Data splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def initialize_models(self):
        """Initialize various regression models with default parameters"""
        logger.info("Initializing models...")
        
        self.models = {
            'linear_regression': LinearRegression(),
            
            'ridge': Ridge(random_state=self.random_state),
            
            'lasso': Lasso(random_state=self.random_state),
            
            'elastic_net': ElasticNet(random_state=self.random_state),
            
            'decision_tree': DecisionTreeRegressor(random_state=self.random_state),
            
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
            
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'svr': SVR(),
            
            'knn': KNeighborsRegressor(n_neighbors=5)
        }
        
        logger.info(f"Initialized {len(self.models)} models")
        return self.models
    
    def get_hyperparameter_grids(self):
        """Define hyperparameter grids for each model"""
        param_grids = {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky']
            },
            
            'lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 2000]
            },
            
            'elastic_net': {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9],
                'max_iter': [1000, 2000]
            },
            
            'decision_tree': {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            
            'extra_trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            
            'svr': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            
            'knn': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            }
        }
        
        return param_grids
    
    def evaluate_model(self, model, X_val, y_val):
        """Evaluate a single model and return metrics"""
        y_pred = model.predict(X_val)
        
        metrics = {
            'mse': mean_squared_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred)
        }
        
        return metrics, y_pred
    
    def cross_validate_models(self, X_train, y_train, cv_folds=5):
        """Perform cross-validation on all models"""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        if not self.models:
            self.initialize_models()
        
        cv_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Cross-validating {name}...")
            
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv_folds, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                cv_results[name] = {
                    'cv_scores': -cv_scores,  # Convert back to positive MSE
                    'mean_cv_score': -cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'rmse_cv': np.sqrt(-cv_scores.mean())
                }
                
                logger.info(f"{name} - CV RMSE: {cv_results[name]['rmse_cv']:.2f} (+/- {cv_scores.std():.2f})")
                
            except Exception as e:
                logger.warning(f"Error in cross-validation for {name}: {str(e)}")
                cv_results[name] = None
        
        self.cv_results = cv_results
        logger.info("Cross-validation completed")
        return cv_results
    
    def train_single_model(self, model_name, X_train, y_train, X_val, y_val):
        """Train a single model and evaluate it"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in initialized models")
        
        model = self.models[model_name]
        logger.info(f"Training {model_name}...")
        
        # Train the model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on training set
        train_metrics, _ = self.evaluate_model(model, X_train, y_train)
        
        # Evaluate on validation set
        val_metrics, val_predictions = self.evaluate_model(model, X_val, y_val)
        
        # Store results
        results = {
            'model': model,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'val_predictions': val_predictions
        }
        
        self.trained_models[model_name] = results
        
        logger.info(f"{model_name} trained - Val RMSE: {val_metrics['rmse']:.2f}, R²: {val_metrics['r2']:.3f}")
        return results
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all initialized models"""
        logger.info("Training all models...")
        
        if not self.models:
            self.initialize_models()
        
        results = {}
        for model_name in self.models.keys():
            try:
                results[model_name] = self.train_single_model(model_name, X_train, y_train, X_val, y_val)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = None
        
        logger.info("All models trained")
        return results
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, method='grid_search', cv_folds=3, n_iter=20):
        """Perform hyperparameter tuning for a specific model"""
        logger.info(f"Tuning hyperparameters for {model_name} using {method}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        param_grids = self.get_hyperparameter_grids()
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return self.models[model_name]
        
        base_model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        try:
            if method == 'grid_search':
                search = GridSearchCV(
                    base_model, param_grid, 
                    cv=cv_folds, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
            elif method == 'random_search':
                search = RandomizedSearchCV(
                    base_model, param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=1
                )
            else:
                raise ValueError(f"Unknown tuning method: {method}")
            
            # Perform the search
            search.fit(X_train, y_train)
            
            # Store results
            tuned_model = search.best_estimator_
            best_params = search.best_params_
            best_score = -search.best_score_  # Convert to positive MSE
            
            logger.info(f"Best parameters for {model_name}: {best_params}")
            logger.info(f"Best CV RMSE: {np.sqrt(best_score):.2f}")
            
            return tuned_model, best_params, best_score
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning for {model_name}: {str(e)}")
            return self.models[model_name], {}, float('inf')
    
    def find_best_model(self, X_val, y_val, metric='rmse'):
        """Find the best model based on validation performance"""
        logger.info(f"Finding best model based on {metric}...")
        
        if not self.trained_models:
            raise ValueError("No trained models found. Train models first.")
        
        best_score = float('inf') if metric in ['mse', 'rmse', 'mae'] else float('-inf')
        best_model_name = None
        
        for name, results in self.trained_models.items():
            if results is None:
                continue
                
            score = results['val_metrics'][metric]
            
            if metric in ['mse', 'rmse', 'mae']:
                if score < best_score:
                    best_score = score
                    best_model_name = name
            else:  # r2 or other metrics where higher is better
                if score > best_score:
                    best_score = score
                    best_model_name = name
        
        if best_model_name:
            self.best_model = self.trained_models[best_model_name]['model']
            logger.info(f"Best model: {best_model_name} with {metric}: {best_score:.3f}")
            return best_model_name, self.best_model, best_score
        else:
            logger.warning("No best model found")
            return None, None, None
    
    def save_model(self, model, filepath, model_info=None):
        """Save trained model to disk"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(model, filepath)
            
            # Save model info if provided
            if model_info:
                info_filepath = filepath.replace('.pkl', '_info.pkl')
                joblib.dump(model_info, info_filepath)
            
            logger.info(f"Model saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        try:
            model = joblib.load(filepath)
            logger.info(f"Model loaded from: {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def model_selection_pipeline(self, filepath, target_column='price', 
                                perform_cv=True, tune_best_models=True,
                                top_models_to_tune=3):
        """Complete model selection and training pipeline"""
        logger.info("Starting model selection and training pipeline...")
        
        # Load data
        df = self.load_engineered_data(filepath)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df, target_column)
        
        # Initialize models
        self.initialize_models()
        
        # Cross-validation
        if perform_cv:
            cv_results = self.cross_validate_models(X_train, y_train)
        
        # Train all models
        training_results = self.train_all_models(X_train, y_train, X_val, y_val)
        
        # Find best model
        best_model_name, best_model, best_score = self.find_best_model(X_val, y_val)
        
        # Hyperparameter tuning for top models
        tuned_models = {}
        if tune_best_models and best_model_name:
            # Get top performing models
            model_scores = []
            for name, results in self.trained_models.items():
                if results is not None:
                    model_scores.append((name, results['val_metrics']['rmse']))
            
            model_scores.sort(key=lambda x: x[1])  # Sort by RMSE (lower is better)
            top_models = [name for name, _ in model_scores[:top_models_to_tune]]
            
            logger.info(f"Tuning top {len(top_models)} models: {top_models}")
            
            for model_name in top_models:
                try:
                    tuned_model, best_params, best_cv_score = self.hyperparameter_tuning(
                        model_name, X_train, y_train, method='grid_search'
                    )
                    
                    # Train tuned model and evaluate
                    tuned_model.fit(X_train, y_train)
                    tuned_metrics, _ = self.evaluate_model(tuned_model, X_val, y_val)
                    
                    tuned_models[model_name] = {
                        'model': tuned_model,
                        'params': best_params,
                        'cv_score': best_cv_score,
                        'val_metrics': tuned_metrics
                    }
                    
                except Exception as e:
                    logger.error(f"Error tuning {model_name}: {str(e)}")
        
        # Find best tuned model
        final_best_model = best_model
        final_best_name = best_model_name
        final_best_score = best_score
        
        if tuned_models:
            for name, results in tuned_models.items():
                if results['val_metrics']['rmse'] < final_best_score:
                    final_best_model = results['model']
                    final_best_name = f"{name}_tuned"
                    final_best_score = results['val_metrics']['rmse']
        
        # Final evaluation on test set
        test_metrics, test_predictions = self.evaluate_model(final_best_model, X_test, y_test)
        
        # Create training report
        training_report = {
            'data_shape': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            },
            'models_trained': list(self.trained_models.keys()),
            'cv_performed': perform_cv,
            'best_model': final_best_name,
            'best_val_score': final_best_score,
            'test_metrics': test_metrics,
            'tuned_models': list(tuned_models.keys()) if tuned_models else [],
            'feature_count': X_train.shape[1]
        }
        
        logger.info(f"Model selection completed. Best model: {final_best_name}")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.2f}, Test R²: {test_metrics['r2']:.3f}")
        
        return {
            'best_model': final_best_model,
            'best_model_name': final_best_name,
            'training_results': training_results,
            'tuned_models': tuned_models,
            'test_data': (X_test, y_test),
            'test_predictions': test_predictions,
            'training_report': training_report
        }

def main():
    """Example usage of the model training module"""
    trainer = ModelTrainer()
    
    # Define file paths
    input_file = "../data/engineered_housing_data.csv"
    model_save_path = "../models/best_model.pkl"
    
    try:
        # Run model selection pipeline
        results = trainer.model_selection_pipeline(
            filepath=input_file,
            target_column='price',
            perform_cv=True,
            tune_best_models=True,
            top_models_to_tune=3
        )
        
        # Save best model
        os.makedirs("../models", exist_ok=True)
        trainer.save_model(results['best_model'], model_save_path, results['training_report'])
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL TRAINING SUMMARY")
        print("="*60)
        report = results['training_report']
        print(f"Best model: {report['best_model']}")
        print(f"Models trained: {len(report['models_trained'])}")
        print(f"Features used: {report['feature_count']}")
        print(f"Test RMSE: {report['test_metrics']['rmse']:.2f}")
        print(f"Test MAE: {report['test_metrics']['mae']:.2f}")
        print(f"Test R²: {report['test_metrics']['r2']:.3f}")
        
        if report['tuned_models']:
            print(f"Tuned models: {', '.join(report['tuned_models'])}")
        
        return results
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
