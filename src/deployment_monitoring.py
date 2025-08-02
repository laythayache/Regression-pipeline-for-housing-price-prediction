"""
Deployment and Monitoring Module
Handles model deployment, monitoring, drift detection, and performance tracking
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime, timedelta
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDeployment:
    def __init__(self, model_path=None, model_metadata_path=None):
        self.model = None
        self.model_metadata = {}
        self.feature_names = []
        self.feature_stats = {}
        self.performance_history = []
        self.deployment_config = {}
        
        if model_path:
            self.load_model(model_path)
        if model_metadata_path:
            self.load_model_metadata(model_metadata_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_model_metadata(self, metadata_path):
        """Load model metadata and training statistics"""
        try:
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            logger.info(f"Model metadata loaded from: {metadata_path}")
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            self.model_metadata = {}
    
    def save_model_package(self, save_dir="../deployment"):
        """Save complete model package for deployment"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "model.pkl")
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2, default=str)
        
        # Save feature statistics
        if self.feature_stats:
            stats_path = os.path.join(save_dir, "feature_stats.pkl")
            with open(stats_path, 'wb') as f:
                pickle.dump(self.feature_stats, f)
        
        # Create deployment config
        deployment_config = {
            "model_version": self.model_metadata.get("version", "1.0"),
            "deployment_date": datetime.now().isoformat(),
            "model_type": str(type(self.model).__name__),
            "feature_count": len(self.feature_names),
            "expected_features": self.feature_names,
            "performance_baseline": self.model_metadata.get("test_metrics", {}),
            "monitoring_enabled": True,
            "drift_detection_enabled": True
        }
        
        config_path = os.path.join(save_dir, "deployment_config.json")
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        self.deployment_config = deployment_config
        logger.info(f"Model package saved to: {save_dir}")
        return save_dir
    
    def validate_input_data(self, data):
        """Validate input data before making predictions"""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check if data is DataFrame
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Data cannot be converted to DataFrame")
                return validation_results, data
        
        # Check feature count
        if len(data.columns) != len(self.feature_names):
            validation_results["warnings"].append(
                f"Feature count mismatch. Expected: {len(self.feature_names)}, Got: {len(data.columns)}"
            )
        
        # Check for missing features
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Missing features: {list(missing_features)}")
        
        # Check for extra features
        extra_features = set(data.columns) - set(self.feature_names)
        if extra_features:
            validation_results["warnings"].append(f"Extra features found: {list(extra_features)}")
            # Remove extra features
            data = data[self.feature_names]
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            validation_results["warnings"].append(f"Missing values detected: {missing_values[missing_values > 0].to_dict()}")
        
        # Check data types
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                validation_results["warnings"].append(f"Non-numeric data type in column: {col}")
        
        return validation_results, data
    
    def predict(self, data, validate_input=True):
        """Make predictions with optional input validation"""
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Validate input if requested
        if validate_input:
            validation_results, data = self.validate_input_data(data)
            
            if not validation_results["is_valid"]:
                raise ValueError(f"Input validation failed: {validation_results['issues']}")
            
            if validation_results["warnings"]:
                for warning in validation_results["warnings"]:
                    logger.warning(warning)
        
        # Make predictions
        try:
            predictions = self.model.predict(data)
            
            # Log prediction made
            self.log_prediction(data, predictions)
            
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def log_prediction(self, input_data, predictions):
        """Log prediction for monitoring purposes"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_shape": input_data.shape,
            "prediction_count": len(predictions),
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions))
            }
        }
        
        # Store in performance history (limit to last 1000 entries)
        self.performance_history.append(log_entry)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

class ModelMonitor:
    def __init__(self, deployment_dir="../deployment", monitoring_dir="../monitoring"):
        self.deployment_dir = deployment_dir
        self.monitoring_dir = monitoring_dir
        self.baseline_stats = {}
        self.performance_metrics = []
        self.drift_alerts = []
        
        os.makedirs(monitoring_dir, exist_ok=True)
    
    def load_baseline_stats(self, training_data_path=None):
        """Load baseline statistics from training data"""
        try:
            # Try to load from deployment package first
            stats_path = os.path.join(self.deployment_dir, "feature_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    self.baseline_stats = pickle.load(f)
                logger.info("Baseline statistics loaded from deployment package")
                return
            
            # If not available, calculate from training data
            if training_data_path and os.path.exists(training_data_path):
                training_data = pd.read_csv(training_data_path)
                self.calculate_baseline_stats(training_data)
                logger.info("Baseline statistics calculated from training data")
            else:
                logger.warning("No baseline statistics available")
                
        except Exception as e:
            logger.error(f"Error loading baseline statistics: {str(e)}")
    
    def calculate_baseline_stats(self, training_data):
        """Calculate baseline statistics from training data"""
        numeric_cols = training_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'price':  # Exclude target variable
                self.baseline_stats[col] = {
                    'mean': float(training_data[col].mean()),
                    'std': float(training_data[col].std()),
                    'min': float(training_data[col].min()),
                    'max': float(training_data[col].max()),
                    'q25': float(training_data[col].quantile(0.25)),
                    'q50': float(training_data[col].quantile(0.50)),
                    'q75': float(training_data[col].quantile(0.75))
                }
        
        # Save baseline stats
        stats_path = os.path.join(self.monitoring_dir, "baseline_stats.pkl")
        with open(stats_path, 'wb') as f:
            pickle.dump(self.baseline_stats, f)
    
    def detect_data_drift(self, new_data, threshold=0.05):
        """Detect data drift using statistical tests"""
        if not self.baseline_stats:
            logger.warning("No baseline statistics available for drift detection")
            return {}
        
        drift_results = {}
        
        for col in new_data.columns:
            if col in self.baseline_stats:
                baseline_stats = self.baseline_stats[col]
                new_col_data = new_data[col].dropna()
                
                if len(new_col_data) == 0:
                    continue
                
                # Calculate current statistics
                current_stats = {
                    'mean': float(new_col_data.mean()),
                    'std': float(new_col_data.std()),
                    'min': float(new_col_data.min()),
                    'max': float(new_col_data.max())
                }
                
                # Statistical tests for drift detection
                drift_detected = False
                drift_tests = {}
                
                # 1. Mean shift detection (z-test)
                if baseline_stats['std'] > 0:
                    z_score = abs(current_stats['mean'] - baseline_stats['mean']) / baseline_stats['std']
                    critical_z = stats.norm.ppf(1 - threshold/2)  # Two-tailed test
                    drift_tests['mean_shift'] = {
                        'z_score': z_score,
                        'critical_value': critical_z,
                        'drift_detected': z_score > critical_z
                    }
                    if z_score > critical_z:
                        drift_detected = True
                
                # 2. Variance change detection
                if baseline_stats['std'] > 0 and current_stats['std'] > 0:
                    variance_ratio = current_stats['std'] / baseline_stats['std']
                    drift_tests['variance_change'] = {
                        'ratio': variance_ratio,
                        'drift_detected': variance_ratio > 2 or variance_ratio < 0.5
                    }
                    if variance_ratio > 2 or variance_ratio < 0.5:
                        drift_detected = True
                
                # 3. Range check
                range_violations = {
                    'below_min': (new_col_data < baseline_stats['min']).sum(),
                    'above_max': (new_col_data > baseline_stats['max']).sum()
                }
                total_violations = range_violations['below_min'] + range_violations['above_max']
                violation_rate = total_violations / len(new_col_data)
                
                drift_tests['range_violations'] = {
                    'violation_rate': violation_rate,
                    'violations': range_violations,
                    'drift_detected': violation_rate > 0.1  # 10% threshold
                }
                if violation_rate > 0.1:
                    drift_detected = True
                
                drift_results[col] = {
                    'drift_detected': drift_detected,
                    'current_stats': current_stats,
                    'baseline_stats': baseline_stats,
                    'tests': drift_tests
                }
        
        # Log drift detection results
        if any(result['drift_detected'] for result in drift_results.values()):
            drift_alert = {
                'timestamp': datetime.now().isoformat(),
                'drift_detected': True,
                'affected_features': [col for col, result in drift_results.items() if result['drift_detected']],
                'details': drift_results
            }
            self.drift_alerts.append(drift_alert)
            logger.warning(f"Data drift detected in features: {drift_alert['affected_features']}")
        
        return drift_results
    
    def evaluate_model_performance(self, model, X_new, y_new, model_name="Current Model"):
        """Evaluate model performance on new data"""
        try:
            # Make predictions
            y_pred = model.predict(X_new)
            
            # Calculate metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'sample_size': len(y_new),
                'rmse': float(np.sqrt(mean_squared_error(y_new, y_pred))),
                'mae': float(mean_absolute_error(y_new, y_pred)),
                'r2': float(r2_score(y_new, y_pred)),
                'mean_prediction': float(np.mean(y_pred)),
                'mean_actual': float(np.mean(y_new))
            }
            
            # Store metrics
            self.performance_metrics.append(metrics)
            
            # Save metrics to file
            metrics_path = os.path.join(self.monitoring_dir, "performance_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            logger.info(f"Performance evaluation completed - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {str(e)}")
            raise
    
    def check_performance_degradation(self, current_metrics, baseline_metrics, threshold=0.1):
        """Check if model performance has degraded significantly"""
        degradation_detected = False
        degradation_details = {}
        
        # Compare key metrics
        for metric in ['rmse', 'mae', 'r2']:
            if metric in current_metrics and metric in baseline_metrics:
                current_val = current_metrics[metric]
                baseline_val = baseline_metrics[metric]
                
                if metric in ['rmse', 'mae']:  # Lower is better
                    degradation = (current_val - baseline_val) / baseline_val
                    degradation_detected = degradation_detected or (degradation > threshold)
                else:  # r2 - higher is better
                    degradation = (baseline_val - current_val) / baseline_val
                    degradation_detected = degradation_detected or (degradation > threshold)
                
                degradation_details[metric] = {
                    'current': current_val,
                    'baseline': baseline_val,
                    'degradation_pct': degradation * 100,
                    'threshold_exceeded': abs(degradation) > threshold
                }
        
        if degradation_detected:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'performance_degradation',
                'details': degradation_details,
                'recommendation': 'Consider model retraining'
            }
            self.drift_alerts.append(alert)
            logger.warning("Model performance degradation detected")
        
        return degradation_detected, degradation_details
    
    def generate_monitoring_report(self, save_report=True):
        """Generate comprehensive monitoring report"""
        report_content = []
        report_content.append("="*80)
        report_content.append("MODEL MONITORING REPORT")
        report_content.append("="*80)
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Performance metrics summary
        if self.performance_metrics:
            report_content.append("PERFORMANCE METRICS SUMMARY")
            report_content.append("-" * 40)
            latest_metrics = self.performance_metrics[-1]
            report_content.append(f"Latest evaluation: {latest_metrics['timestamp']}")
            report_content.append(f"Sample size: {latest_metrics['sample_size']}")
            report_content.append(f"RMSE: {latest_metrics['rmse']:.2f}")
            report_content.append(f"MAE: {latest_metrics['mae']:.2f}")
            report_content.append(f"R²: {latest_metrics['r2']:.3f}")
            report_content.append("")
        
        # Drift alerts summary
        if self.drift_alerts:
            report_content.append("DRIFT ALERTS SUMMARY")
            report_content.append("-" * 40)
            report_content.append(f"Total alerts: {len(self.drift_alerts)}")
            
            # Recent alerts (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent_alerts = [
                alert for alert in self.drift_alerts 
                if datetime.fromisoformat(alert['timestamp']) > week_ago
            ]
            
            if recent_alerts:
                report_content.append(f"Recent alerts (last 7 days): {len(recent_alerts)}")
                for alert in recent_alerts[-5:]:  # Show last 5
                    report_content.append(f"  - {alert['timestamp']}: {alert.get('alert_type', 'drift')}")
            else:
                report_content.append("No recent alerts")
            report_content.append("")
        
        # Recommendations
        report_content.append("RECOMMENDATIONS")
        report_content.append("-" * 40)
        
        if len(self.drift_alerts) > 10:
            report_content.append("• High number of drift alerts detected. Consider model retraining.")
        
        if self.performance_metrics and len(self.performance_metrics) > 1:
            latest = self.performance_metrics[-1]
            previous = self.performance_metrics[-2]
            rmse_change = (latest['rmse'] - previous['rmse']) / previous['rmse']
            
            if rmse_change > 0.1:
                report_content.append("• Model performance declining. Review recent data quality.")
        
        if not self.performance_metrics:
            report_content.append("• No performance metrics available. Set up regular evaluation.")
        
        if not self.baseline_stats:
            report_content.append("• No baseline statistics available. Configure drift detection.")
        
        # Save report
        if save_report:
            report_path = os.path.join(self.monitoring_dir, f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_content))
            logger.info(f"Monitoring report saved to: {report_path}")
        
        return '\n'.join(report_content)

def main():
    """Example usage of the deployment and monitoring module"""
    
    # Example deployment workflow
    try:
        # 1. Deploy model
        deployment = ModelDeployment("../models/best_model.pkl")
        
        # Set feature names (normally these would come from training)
        deployment.feature_names = ['sqft_living', 'bedrooms', 'bathrooms', 'age', 
                                  'sqft_lot', 'floors', 'condition', 'grade']
        
        # Save deployment package
        deployment_dir = deployment.save_model_package()
        
        # 2. Set up monitoring
        monitor = ModelMonitor(deployment_dir)
        
        # Load baseline stats (normally from training data)
        monitor.load_baseline_stats("../data/engineered_housing_data.csv")
        
        # 3. Example monitoring workflow
        # Simulate new data for monitoring
        new_data = pd.DataFrame({
            'sqft_living': np.random.normal(2000, 800, 100),
            'bedrooms': np.random.choice([2, 3, 4], 100),
            'bathrooms': np.random.choice([2.0, 2.5, 3.0], 100),
            'age': np.random.exponential(15, 100),
            'sqft_lot': np.random.lognormal(9, 0.8, 100),
            'floors': np.random.choice([1, 2], 100),
            'condition': np.random.choice([3, 4, 5], 100),
            'grade': np.random.choice([6, 7, 8], 100)
        })
        
        # Check for data drift
        drift_results = monitor.detect_data_drift(new_data)
        
        # Generate monitoring report
        report = monitor.generate_monitoring_report()
        
        print("\n" + "="*60)
        print("DEPLOYMENT & MONITORING SUMMARY")
        print("="*60)
        print(f"Model deployed to: {deployment_dir}")
        print(f"Features monitored: {len(deployment.feature_names)}")
        print(f"Drift detection results: {len(drift_results)} features analyzed")
        print(f"Drift alerts: {len(monitor.drift_alerts)}")
        
        if drift_results:
            drift_detected = any(result['drift_detected'] for result in drift_results.values())
            print(f"Drift detected: {'Yes' if drift_detected else 'No'}")
        
        return deployment, monitor
        
    except Exception as e:
        logger.error(f"Deployment/monitoring failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
