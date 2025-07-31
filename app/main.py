"""
Main Pipeline Orchestrator
Coordinates the entire housing price prediction pipeline from data generation to deployment
"""

import sys
import os
import json
import logging
from datetime import datetime
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import pipeline modules
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_selection_training import ModelTrainer
from evaluation import ModelEvaluator
from deployment_monitoring import ModelDeployment, ModelMonitor

# Configure basic logging first (will be reconfigured later)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HousingPricePipeline:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.results = {}
        self.pipeline_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
        # Create directories
        self.setup_directories()
        
        logger.info(f"Pipeline initialized with ID: {self.pipeline_id}")
    
    def load_config(self, config_path):
        """Load pipeline configuration"""
        default_config = {
            "data": {
                "raw_data_path": "../data/housing_data.csv",
                "processed_data_path": "../data/processed_housing_data.csv",
                "engineered_data_path": "../data/engineered_housing_data.csv",
                "target_column": "price"
            },
            "preprocessing": {
                "missing_strategy": "mean",
                "remove_duplicates": True,
                "handle_outliers": True,
                "outlier_method": "iqr",
                "scale_method": "standard"
            },
            "feature_engineering": {
                "create_domain_features": True,
                "create_interactions": True,
                "create_polynomial": False,
                "create_binned": True,
                "feature_selection_method": "k_best",
                "k_features": 15,
                "apply_pca": False
            },
            "model_training": {
                "perform_cv": True,
                "tune_best_models": True,
                "top_models_to_tune": 3,
                "test_size": 0.2,
                "val_size": 0.1
            },
            "evaluation": {
                "save_plots": True,
                "create_comparison": True,
                "generate_report": True
            },
            "deployment": {
                "save_model_package": True,
                "enable_monitoring": True,
                "deployment_dir": "../deployment",
                "monitoring_dir": "../monitoring"
            },
            "output": {
                "models_dir": "../models",
                "plots_dir": "../plots",
                "reports_dir": "../reports",
                "logs_dir": "../logs"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Update default config with user config
                self.update_nested_dict(default_config, user_config)
                logger.info(f"Configuration loaded from: {config_path}")
            except Exception as e:
                logger.warning(f"Error loading config file: {str(e)}. Using default configuration.")
        
        return default_config
    
    def update_nested_dict(self, base_dict, update_dict):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self.update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config["output"]["models_dir"],
            self.config["output"]["plots_dir"], 
            self.config["output"]["reports_dir"],
            self.config["output"]["logs_dir"],
            self.config["deployment"]["deployment_dir"],
            self.config["deployment"]["monitoring_dir"]
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Now configure file logging
        self.setup_logging()
        
        logger.info("Directory structure created")
    
    def setup_logging(self):
        """Setup file logging after directories are created"""
        try:
            log_file = os.path.join(self.config["output"]["logs_dir"], "pipeline.log")
            
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
            logger.info(f"File logging configured: {log_file}")
        except Exception as e:
            logger.warning(f"Could not setup file logging: {str(e)}")
    
    def run_preprocessing(self):
        """Execute preprocessing step"""
        logger.info("="*60)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("="*60)
        
        try:
            processed_df, initial_quality, final_quality, cleaning_report = self.preprocessor.preprocess_pipeline(
                filepath=self.config["data"]["raw_data_path"],
                missing_strategy=self.config["preprocessing"]["missing_strategy"],
                remove_duplicates=self.config["preprocessing"]["remove_duplicates"],
                handle_outliers=self.config["preprocessing"]["handle_outliers"],
                outlier_method=self.config["preprocessing"]["outlier_method"],
                scale_method=self.config["preprocessing"]["scale_method"],
                target_column=self.config["data"]["target_column"]
            )
            
            # Save processed data
            self.preprocessor.save_processed_data(processed_df, self.config["data"]["processed_data_path"])
            
            # Store results
            self.results["preprocessing"] = {
                "status": "success",
                "initial_quality": initial_quality,
                "final_quality": final_quality,
                "cleaning_report": cleaning_report,
                "output_path": self.config["data"]["processed_data_path"]
            }
            
            logger.info(f"[SUCCESS] Preprocessing completed successfully")
            logger.info(f"  Original shape: {cleaning_report['original_shape']}")
            logger.info(f"  Final shape: {cleaning_report['final_shape']}")
            logger.info(f"  Rows removed: {cleaning_report['rows_removed']}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Preprocessing failed: {str(e)}")
            self.results["preprocessing"] = {"status": "failed", "error": str(e)}
            return False
    
    def run_feature_engineering(self):
        """Execute feature engineering step"""
        logger.info("="*60)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*60)
        
        try:
            final_df, X, y, fe_report = self.feature_engineer.feature_engineering_pipeline(
                filepath=self.config["data"]["processed_data_path"],
                target_column=self.config["data"]["target_column"],
                create_domain=self.config["feature_engineering"]["create_domain_features"],
                create_interactions=self.config["feature_engineering"]["create_interactions"],
                create_polynomial=self.config["feature_engineering"]["create_polynomial"],
                create_binned=self.config["feature_engineering"]["create_binned"],
                feature_selection_method=self.config["feature_engineering"]["feature_selection_method"],
                k_features=self.config["feature_engineering"]["k_features"],
                apply_pca_flag=self.config["feature_engineering"]["apply_pca"]
            )
            
            # Save engineered data
            self.feature_engineer.save_engineered_data(final_df, self.config["data"]["engineered_data_path"])
            
            # Store results
            self.results["feature_engineering"] = {
                "status": "success",
                "fe_report": fe_report,
                "output_path": self.config["data"]["engineered_data_path"],
                "feature_names": list(X.columns)
            }
            
            logger.info(f"[SUCCESS] Feature engineering completed successfully")
            logger.info(f"  Original features: {fe_report['original_features']}")
            logger.info(f"  Engineered features created: {fe_report['engineered_features_created']}")
            logger.info(f"  Final features: {fe_report['final_features']}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Feature engineering failed: {str(e)}")
            self.results["feature_engineering"] = {"status": "failed", "error": str(e)}
            return False
    
    def run_model_training(self):
        """Execute model selection and training step"""
        logger.info("="*60)
        logger.info("STEP 3: MODEL SELECTION & TRAINING")
        logger.info("="*60)
        
        try:
            training_results = self.model_trainer.model_selection_pipeline(
                filepath=self.config["data"]["engineered_data_path"],
                target_column=self.config["data"]["target_column"],
                perform_cv=self.config["model_training"]["perform_cv"],
                tune_best_models=self.config["model_training"]["tune_best_models"],
                top_models_to_tune=self.config["model_training"]["top_models_to_tune"]
            )
            
            # Save best model
            model_path = os.path.join(self.config["output"]["models_dir"], f"best_model_{self.pipeline_id}.pkl")
            self.model_trainer.save_model(
                training_results['best_model'], 
                model_path, 
                training_results['training_report']
            )
            
            # Store results
            self.results["model_training"] = {
                "status": "success",
                "best_model": training_results['best_model_name'],
                "training_report": training_results['training_report'],
                "model_path": model_path,
                "test_data": training_results['test_data'],
                "test_predictions": training_results['test_predictions']
            }
            
            logger.info(f"[SUCCESS] Model training completed successfully")
            logger.info(f"  Best model: {training_results['best_model_name']}")
            logger.info(f"  Test RMSE: {training_results['training_report']['test_metrics']['rmse']:.2f}")
            logger.info(f"  Test R²: {training_results['training_report']['test_metrics']['r2']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Model training failed: {str(e)}")
            self.results["model_training"] = {"status": "failed", "error": str(e)}
            return False
    
    def run_evaluation(self):
        """Execute model evaluation step"""
        logger.info("="*60)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("="*60)
        
        try:
            # Get test data and model from training results
            X_test, y_test = self.results["model_training"]["test_data"]
            model_path = self.results["model_training"]["model_path"]
            best_model = self.evaluator.load_model(model_path)
            feature_names = self.results["feature_engineering"]["feature_names"]
            
            # Comprehensive evaluation
            evaluation_results = self.evaluator.evaluate_model_comprehensive(
                model=best_model,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                model_name=self.results["model_training"]["best_model"],
                save_plots=self.config["evaluation"]["save_plots"],
                plot_dir=self.config["output"]["plots_dir"]
            )
            
            # Generate evaluation report
            if self.config["evaluation"]["generate_report"]:
                report = self.evaluator.generate_evaluation_report(
                    save_report=True,
                    report_dir=self.config["output"]["reports_dir"]
                )
            
            # Store results
            self.results["evaluation"] = {
                "status": "success",
                "evaluation_results": evaluation_results,
                "plots_created": len(self.evaluator.plots_saved),
                "report_generated": self.config["evaluation"]["generate_report"]
            }
            
            logger.info(f"[SUCCESS] Model evaluation completed successfully")
            logger.info(f"  RMSE: {evaluation_results['metrics']['rmse']:.2f}")
            logger.info(f"  R²: {evaluation_results['metrics']['r2']:.3f}")
            logger.info(f"  Plots created: {len(self.evaluator.plots_saved)}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Model evaluation failed: {str(e)}")
            self.results["evaluation"] = {"status": "failed", "error": str(e)}
            return False
    
    def run_deployment(self):
        """Execute deployment and monitoring setup step"""
        logger.info("="*60)
        logger.info("STEP 5: DEPLOYMENT & MONITORING")
        logger.info("="*60)
        
        try:
            # Set up deployment
            model_path = self.results["model_training"]["model_path"]
            deployment = ModelDeployment(model_path)
            
            # Set feature names
            deployment.feature_names = self.results["feature_engineering"]["feature_names"]
            deployment.model_metadata = self.results["model_training"]["training_report"]
            
            # Save deployment package
            if self.config["deployment"]["save_model_package"]:
                deployment_dir = deployment.save_model_package(self.config["deployment"]["deployment_dir"])
            
            # Set up monitoring
            if self.config["deployment"]["enable_monitoring"]:
                monitor = ModelMonitor(
                    deployment_dir=self.config["deployment"]["deployment_dir"],
                    monitoring_dir=self.config["deployment"]["monitoring_dir"]
                )
                
                # Load baseline statistics
                monitor.load_baseline_stats(self.config["data"]["engineered_data_path"])
                
                # Generate initial monitoring report
                monitoring_report = monitor.generate_monitoring_report()
            
            # Store results
            self.results["deployment"] = {
                "status": "success",
                "deployment_dir": self.config["deployment"]["deployment_dir"],
                "monitoring_enabled": self.config["deployment"]["enable_monitoring"],
                "monitoring_dir": self.config["deployment"]["monitoring_dir"]
            }
            
            logger.info(f"[SUCCESS] Deployment completed successfully")
            logger.info(f"  Model deployed to: {self.config['deployment']['deployment_dir']}")
            logger.info(f"  Monitoring enabled: {self.config['deployment']['enable_monitoring']}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Deployment failed: {str(e)}")
            self.results["deployment"] = {"status": "failed", "error": str(e)}
            return False
    
    def save_pipeline_results(self):
        """Save complete pipeline results"""
        try:
            results_path = os.path.join(self.config["output"]["reports_dir"], f"pipeline_results_{self.pipeline_id}.json")
            
            # Convert numpy arrays and other non-serializable objects to serializable format
            serializable_results = self.make_serializable(self.results.copy())
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Pipeline results saved to: {results_path}")
            return results_path
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {str(e)}")
            return None
    
    def make_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self.make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return str(obj)
        else:
            return obj
    
    def generate_pipeline_summary(self):
        """Generate summary of pipeline execution"""
        summary = []
        summary.append("="*80)
        summary.append("HOUSING PRICE PREDICTION PIPELINE SUMMARY")
        summary.append("="*80)
        summary.append(f"Pipeline ID: {self.pipeline_id}")
        summary.append(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # Step-by-step summary
        steps = [
            ("preprocessing", "Data Preprocessing"),
            ("feature_engineering", "Feature Engineering"),
            ("model_training", "Model Training"),
            ("evaluation", "Model Evaluation"),
            ("deployment", "Deployment & Monitoring")
        ]
        
        for step_key, step_name in steps:
            if step_key in self.results:
                status = self.results[step_key]["status"]
                status_symbol = "[SUCCESS]" if status == "success" else "[FAILED]"
                summary.append(f"{status_symbol} {step_name}: {status.upper()}")
                
                if status == "success":
                    # Add step-specific details
                    if step_key == "preprocessing":
                        report = self.results[step_key]["cleaning_report"]
                        summary.append(f"    Rows processed: {report['original_shape'][0]} -> {report['final_shape'][0]}")
                        
                    elif step_key == "feature_engineering":
                        report = self.results[step_key]["fe_report"]
                        summary.append(f"    Features: {report['original_features']} -> {report['final_features']}")
                        
                    elif step_key == "model_training":
                        summary.append(f"    Best model: {self.results[step_key]['best_model']}")
                        metrics = self.results[step_key]["training_report"]["test_metrics"]
                        summary.append(f"    Test RMSE: {metrics['rmse']:.2f}")
                        summary.append(f"    Test R²: {metrics['r2']:.3f}")
                        
                    elif step_key == "evaluation":
                        summary.append(f"    Plots created: {self.results[step_key]['plots_created']}")
                        
                    elif step_key == "deployment":
                        summary.append(f"    Deployed to: {self.results[step_key]['deployment_dir']}")
                else:
                    summary.append(f"    Error: {self.results[step_key].get('error', 'Unknown error')}")
            else:
                summary.append(f"[NOT EXECUTED] {step_name}: NOT EXECUTED")
            
            summary.append("")
        
        # Overall success rate
        total_steps = len(steps)
        successful_steps = sum(1 for step_key, _ in steps 
                             if step_key in self.results and self.results[step_key]["status"] == "success")
        success_rate = (successful_steps / total_steps) * 100
        
        summary.append(f"Overall Success Rate: {success_rate:.1f}% ({successful_steps}/{total_steps} steps)")
        summary.append("")
        
        # File outputs
        summary.append("OUTPUT FILES:")
        summary.append("-" * 20)
        if "preprocessing" in self.results and self.results["preprocessing"]["status"] == "success":
            summary.append(f"• Processed data: {self.config['data']['processed_data_path']}")
        if "feature_engineering" in self.results and self.results["feature_engineering"]["status"] == "success":
            summary.append(f"• Engineered data: {self.config['data']['engineered_data_path']}")
        if "model_training" in self.results and self.results["model_training"]["status"] == "success":
            summary.append(f"• Best model: {self.results['model_training']['model_path']}")
        if "deployment" in self.results and self.results["deployment"]["status"] == "success":
            summary.append(f"• Deployment package: {self.results['deployment']['deployment_dir']}")
        
        return "\n".join(summary)
    
    def run_complete_pipeline(self):
        """Execute the complete pipeline"""
        logger.info("Starting complete housing price prediction pipeline...")
        start_time = datetime.now()
        
        try:
            # Execute pipeline steps
            steps_success = []
            
            # Step 1: Preprocessing
            steps_success.append(self.run_preprocessing())
            
            # Step 2: Feature Engineering (only if preprocessing succeeded)
            if steps_success[-1]:
                steps_success.append(self.run_feature_engineering())
            else:
                logger.error("Skipping feature engineering due to preprocessing failure")
                steps_success.append(False)
            
            # Step 3: Model Training (only if feature engineering succeeded)
            if steps_success[-1]:
                steps_success.append(self.run_model_training())
            else:
                logger.error("Skipping model training due to feature engineering failure")
                steps_success.append(False)
            
            # Step 4: Evaluation (only if training succeeded)
            if steps_success[-1]:
                steps_success.append(self.run_evaluation())
            else:
                logger.error("Skipping evaluation due to model training failure")
                steps_success.append(False)
            
            # Step 5: Deployment (only if evaluation succeeded)
            if steps_success[-1]:
                steps_success.append(self.run_deployment())
            else:
                logger.error("Skipping deployment due to evaluation failure")
                steps_success.append(False)
            
            # Save results and generate summary
            self.save_pipeline_results()
            summary = self.generate_pipeline_summary()
            
            # Calculate execution time
            execution_time = datetime.now() - start_time
            
            logger.info("="*60)
            logger.info("PIPELINE EXECUTION COMPLETED")
            logger.info("="*60)
            logger.info(f"Total execution time: {execution_time}")
            logger.info(f"Successful steps: {sum(steps_success)}/{len(steps_success)}")
            
            # Print summary
            print(summary)
            
            # Save summary to file
            summary_path = os.path.join(self.config["output"]["reports_dir"], f"pipeline_summary_{self.pipeline_id}.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            return sum(steps_success) == len(steps_success), self.results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False, self.results

def main():
    """Main function to run the pipeline"""
    print("="*80)
    print("HOUSING PRICE PREDICTION PIPELINE")
    print("="*80)
    print("Starting automated machine learning pipeline...")
    print()
    
    try:
        # Initialize and run pipeline
        pipeline = HousingPricePipeline()
        success, results = pipeline.run_complete_pipeline()
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            print(f"Check the results in the '{pipeline.config['output']['reports_dir']}' directory")
        else:
            print("\n❌ Pipeline completed with errors.")
            print("Check the logs for detailed error information.")
        
        return pipeline, results
        
    except Exception as e:
        print(f"\n💥 Pipeline failed to start: {str(e)}")
        print("Check the configuration and try again.")
        raise

if __name__ == "__main__":
    pipeline, results = main()
