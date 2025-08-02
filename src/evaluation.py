"""
Model Evaluation Module
Handles comprehensive model evaluation, metrics calculation, visualization, and performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           explained_variance_score, max_error, mean_absolute_percentage_error)
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        self.plots_saved = []
        
    def load_model(self, filepath):
        """Load trained model from disk"""
        try:
            model = joblib.load(filepath)
            logger.info(f"Model loaded from: {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def calculate_regression_metrics(self, y_true, y_pred):
        """Calculate comprehensive regression metrics"""
        metrics = {
            # Basic metrics
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            
            # Additional metrics
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred),
            
            # Percentage errors
            'mean_error': np.mean(y_pred - y_true),
            'mean_abs_percentage_error': mean_absolute_percentage_error(y_true, y_pred) * 100,
            
            # Custom metrics
            'residual_std': np.std(y_pred - y_true),
            'mean_residual': np.mean(y_pred - y_true),
        }
        
        # Add percentage accuracy (within certain error ranges)
        error_percentages = np.abs((y_pred - y_true) / y_true) * 100
        metrics['accuracy_within_5pct'] = np.mean(error_percentages <= 5) * 100
        metrics['accuracy_within_10pct'] = np.mean(error_percentages <= 10) * 100
        metrics['accuracy_within_20pct'] = np.mean(error_percentages <= 20) * 100
        
        return metrics
    
    def create_prediction_plots(self, y_true, y_pred, model_name="Model", save_dir="../plots"):
        """Create various prediction visualization plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up the plot style
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue', s=30)
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'{model_name}: Actual vs Predicted Values', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        plt.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plot_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_actual_vs_predicted.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_saved.append(plot_path)
        
        # 2. Residuals plot
        residuals = y_pred - y_true
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals (Predicted - Actual)', fontsize=12)
        plt.title(f'{model_name}: Residuals Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_residuals.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_saved.append(plot_path)
        
        # 3. Residuals distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Residuals', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'{model_name}: Distribution of Residuals', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add normal distribution curve
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * 
             np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))
        y = y * len(residuals) * (residuals.max() - residuals.min()) / 50  # Scale to histogram
        plt.plot(x, y, 'orange', linewidth=2, label=f'Normal (μ={mu:.1f}, σ={sigma:.1f})')
        plt.legend()
        
        plot_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_residuals_dist.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_saved.append(plot_path)
        
        # 4. Q-Q plot for residuals normality
        from scipy import stats
        plt.figure(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{model_name}: Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_qq_plot.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_saved.append(plot_path)
        
        logger.info(f"Created 4 prediction plots for {model_name}")
    
    def create_error_analysis_plots(self, y_true, y_pred, model_name="Model", save_dir="../plots"):
        """Create error analysis visualization plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate error metrics
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        percentage_errors = np.abs(errors / y_true) * 100
        
        # 1. Error distribution by prediction magnitude
        plt.figure(figsize=(12, 8))
        
        # Create bins based on predicted values
        pred_bins = pd.qcut(y_pred, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        plt.subplot(2, 2, 1)
        boxplot_data = [abs_errors[pred_bins == bin_name] for bin_name in pred_bins.categories]
        plt.boxplot(boxplot_data, labels=pred_bins.categories)
        plt.title('Absolute Error by Prediction Range')
        plt.ylabel('Absolute Error')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        boxplot_data = [percentage_errors[pred_bins == bin_name] for bin_name in pred_bins.categories]
        plt.boxplot(boxplot_data, labels=pred_bins.categories)
        plt.title('Percentage Error by Prediction Range')
        plt.ylabel('Percentage Error (%)')
        plt.xticks(rotation=45)
        
        # 3. Error vs actual values
        plt.subplot(2, 2, 3)
        plt.scatter(y_true, abs_errors, alpha=0.6, color='red', s=20)
        plt.xlabel('Actual Values')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Error vs Actual Values')
        plt.grid(True, alpha=0.3)
        
        # 4. Percentage error distribution
        plt.subplot(2, 2, 4)
        plt.hist(percentage_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Percentage Error (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Percentage Errors')
        plt.axvline(x=np.median(percentage_errors), color='red', linestyle='--', 
                   label=f'Median: {np.median(percentage_errors):.1f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name}: Error Analysis', fontsize=16, fontweight='bold')
        plot_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_error_analysis.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_saved.append(plot_path)
        
        logger.info(f"Created error analysis plot for {model_name}")
    
    def create_feature_importance_plot(self, model, feature_names, model_name="Model", save_dir="../plots"):
        """Create feature importance plot if model supports it"""
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Check if model has feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                logger.warning(f"Model {model_name} doesn't support feature importance")
                return
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            # Plot top 20 features
            top_features = feature_importance_df.tail(20)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name}: Top 20 Feature Importances', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            
            plot_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.plots_saved.append(plot_path)
            
            logger.info(f"Created feature importance plot for {model_name}")
            return feature_importance_df
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            return None
    
    def create_learning_curves(self, model, X, y, model_name="Model", save_dir="../plots"):
        """Create learning curves to analyze model performance vs training size"""
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            logger.info(f"Generating learning curves for {model_name}...")
            
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='neg_mean_squared_error',
                random_state=42
            )
            
            # Convert to RMSE
            train_rmse = np.sqrt(-train_scores)
            val_rmse = np.sqrt(-val_scores)
            
            # Calculate means and standard deviations
            train_rmse_mean = np.mean(train_rmse, axis=1)
            train_rmse_std = np.std(train_rmse, axis=1)
            val_rmse_mean = np.mean(val_rmse, axis=1)
            val_rmse_std = np.std(val_rmse, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_rmse_mean, 'o-', color='blue', label='Training RMSE')
            plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std,
                           train_rmse_mean + train_rmse_std, alpha=0.1, color='blue')
            
            plt.plot(train_sizes, val_rmse_mean, 'o-', color='red', label='Validation RMSE')
            plt.fill_between(train_sizes, val_rmse_mean - val_rmse_std,
                           val_rmse_mean + val_rmse_std, alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('RMSE')
            plt.title(f'{model_name}: Learning Curves', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_learning_curves.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.plots_saved.append(plot_path)
            
            logger.info(f"Created learning curves for {model_name}")
            
        except Exception as e:
            logger.error(f"Error creating learning curves: {str(e)}")
    
    def evaluate_model_comprehensive(self, model, X_test, y_test, feature_names=None, 
                                   model_name="Model", save_plots=True, plot_dir="../plots"):
        """Perform comprehensive model evaluation"""
        logger.info(f"Starting comprehensive evaluation of {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_regression_metrics(y_test, y_pred)
        
        # Create visualizations if requested
        if save_plots:
            self.create_prediction_plots(y_test, y_pred, model_name, plot_dir)
            self.create_error_analysis_plots(y_test, y_pred, model_name, plot_dir)
            
            if feature_names is not None:
                feature_importance_df = self.create_feature_importance_plot(
                    model, feature_names, model_name, plot_dir
                )
            else:
                feature_importance_df = None
            
            # Create learning curves (using subset of data for speed)
            if len(X_test) > 100:
                sample_indices = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
                X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
                y_sample = y_test.iloc[sample_indices] if hasattr(y_test, 'iloc') else y_test[sample_indices]
                self.create_learning_curves(model, X_sample, y_sample, model_name, plot_dir)
        
        # Store evaluation results
        evaluation_result = {
            'model_name': model_name,
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test,
            'feature_importance': feature_importance_df.to_dict() if feature_importance_df is not None else None,
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_size': len(y_test)
        }
        
        self.evaluation_results[model_name] = evaluation_result
        
        logger.info(f"Comprehensive evaluation completed for {model_name}")
        return evaluation_result
    
    def compare_models(self, model_results_dict, save_comparison=True, plot_dir="../plots"):
        """Compare multiple models side by side"""
        logger.info("Comparing multiple models...")
        
        if len(model_results_dict) < 2:
            logger.warning("Need at least 2 models for comparison")
            return
        
        # Extract metrics for comparison
        comparison_metrics = {}
        model_names = list(model_results_dict.keys())
        
        # Get all metric names from first model
        first_model = list(model_results_dict.values())[0]
        metric_names = list(first_model['metrics'].keys())
        
        for metric in metric_names:
            comparison_metrics[metric] = []
            for model_name in model_names:
                comparison_metrics[metric].append(model_results_dict[model_name]['metrics'][metric])
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_metrics, index=model_names)
        
        if save_comparison:
            os.makedirs(plot_dir, exist_ok=True)
            
            # 1. Metrics comparison bar plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Key metrics to plot
            key_metrics = ['rmse', 'mae', 'r2', 'mean_abs_percentage_error']
            
            for i, metric in enumerate(key_metrics):
                ax = axes[i//2, i%2]
                if metric in comparison_df.columns:
                    bars = ax.bar(model_names, comparison_df[metric], alpha=0.7)
                    ax.set_title(f'{metric.upper()}', fontweight='bold')
                    ax.set_ylabel(metric.upper())
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
            
            plt.suptitle('Model Comparison: Key Metrics', fontsize=16, fontweight='bold')
            plot_path = os.path.join(plot_dir, 'model_comparison_metrics.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.plots_saved.append(plot_path)
            
            # 2. Detailed metrics heatmap
            plt.figure(figsize=(12, 8))
            
            # Select relevant metrics for heatmap
            heatmap_metrics = ['rmse', 'mae', 'r2', 'explained_variance', 
                             'mean_abs_percentage_error', 'accuracy_within_10pct']
            available_metrics = [m for m in heatmap_metrics if m in comparison_df.columns]
            
            heatmap_data = comparison_df[available_metrics].T
            
            # Normalize data for better visualization (except R2 which is already 0-1)
            normalized_data = heatmap_data.copy()
            for metric in available_metrics:
                if metric not in ['r2', 'explained_variance', 'accuracy_within_5pct', 
                                'accuracy_within_10pct', 'accuracy_within_20pct']:
                    # For error metrics, normalize by max value
                    max_val = heatmap_data.loc[metric].max()
                    if max_val > 0:
                        normalized_data.loc[metric] = heatmap_data.loc[metric] / max_val
            
            sns.heatmap(normalized_data, annot=heatmap_data, fmt='.3f', cmap='RdYlBu_r',
                       center=0.5, square=True, cbar_kws={'label': 'Normalized Score'})
            plt.title('Model Comparison: Detailed Metrics Heatmap', fontsize=14, fontweight='bold')
            plt.ylabel('Metrics')
            plt.xlabel('Models')
            
            plot_path = os.path.join(plot_dir, 'model_comparison_heatmap.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.plots_saved.append(plot_path)
        
        logger.info(f"Model comparison completed for {len(model_names)} models")
        return comparison_df
    
    def generate_evaluation_report(self, evaluation_results=None, save_report=True, report_dir="../reports"):
        """Generate comprehensive evaluation report"""
        if evaluation_results is None:
            evaluation_results = self.evaluation_results
        
        if not evaluation_results:
            logger.warning("No evaluation results to report")
            return
        
        logger.info("Generating evaluation report...")
        
        # Create report content
        report_content = []
        report_content.append("="*80)
        report_content.append("MODEL EVALUATION REPORT")
        report_content.append("="*80)
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"Number of models evaluated: {len(evaluation_results)}")
        report_content.append("")
        
        # Individual model results
        for model_name, results in evaluation_results.items():
            report_content.append(f"Model: {model_name}")
            report_content.append("-" * 50)
            
            metrics = results['metrics']
            report_content.append(f"Test samples: {results['test_size']}")
            report_content.append(f"RMSE: {metrics['rmse']:.2f}")
            report_content.append(f"MAE: {metrics['mae']:.2f}")
            report_content.append(f"R²: {metrics['r2']:.3f}")
            report_content.append(f"Mean Absolute Percentage Error: {metrics['mean_abs_percentage_error']:.2f}%")
            report_content.append(f"Accuracy within 10%: {metrics['accuracy_within_10pct']:.1f}%")
            report_content.append("")
        
        # Model ranking
        if len(evaluation_results) > 1:
            report_content.append("MODEL RANKING")
            report_content.append("-" * 30)
            
            # Rank by RMSE (lower is better)
            models_by_rmse = sorted(evaluation_results.items(), 
                                  key=lambda x: x[1]['metrics']['rmse'])
            
            report_content.append("By RMSE (lower is better):")
            for i, (model_name, results) in enumerate(models_by_rmse, 1):
                rmse = results['metrics']['rmse']
                report_content.append(f"{i}. {model_name}: {rmse:.2f}")
            
            report_content.append("")
            
            # Rank by R² (higher is better)
            models_by_r2 = sorted(evaluation_results.items(), 
                                key=lambda x: x[1]['metrics']['r2'], reverse=True)
            
            report_content.append("By R² (higher is better):")
            for i, (model_name, results) in enumerate(models_by_r2, 1):
                r2 = results['metrics']['r2']
                report_content.append(f"{i}. {model_name}: {r2:.3f}")
        
        # Save report
        if save_report:
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_content))
            
            logger.info(f"Evaluation report saved to: {report_path}")
        
        # Also return as string
        return '\n'.join(report_content)

def main():
    """Example usage of the evaluation module"""
    evaluator = ModelEvaluator()
    
    # Define file paths
    model_path = "../models/best_model.pkl"
    data_path = "../data/engineered_housing_data.csv"
    
    try:
        # Load model and data (this would typically come from the training module)
        model = evaluator.load_model(model_path)
        
        # Load test data (normally you'd have this separated)
        df = pd.read_csv(data_path)
        X = df.drop(columns=['price'])
        y = df['price']
        
        # Use last 20% as test set for demonstration
        split_idx = int(len(df) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # Comprehensive evaluation
        results = evaluator.evaluate_model_comprehensive(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=X.columns.tolist(),
            model_name="Best Model",
            save_plots=True
        )
        
        # Generate report
        report = evaluator.generate_evaluation_report()
        print(report)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model evaluated: {results['model_name']}")
        print(f"Test samples: {results['test_size']}")
        print(f"RMSE: {results['metrics']['rmse']:.2f}")
        print(f"R²: {results['metrics']['r2']:.3f}")
        print(f"Plots created: {len(evaluator.plots_saved)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
