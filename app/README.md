# Housing Price Prediction Pipeline - Application

This directory contains the main application that orchestrates the complete housing price prediction pipeline.

## Overview

The `main.py` script coordinates all pipeline steps:
1. **Preprocessing** - Data cleaning and preparation
2. **Feature Engineering** - Feature creation and selection  
3. **Model Selection & Training** - Model comparison and training
4. **Evaluation** - Comprehensive model evaluation
5. **Deployment & Monitoring** - Model deployment and monitoring setup

## Quick Start

### 1. Install Dependencies
```bash
pip install -r ../requirements.txt
```

### 2. Generate Data (if not already done)
```bash
cd ..
python data.py
```

### 3. Run Complete Pipeline
```bash
cd app
python main.py
```

## Configuration

The pipeline can be customized using `config.json`. Key configuration options:

### Data Configuration
- `target_column`: Target variable name (default: "price")
- Data file paths for each pipeline stage

### Preprocessing Options
- `missing_strategy`: "mean", "median", "mode", "knn", "drop_rows"
- `handle_outliers`: Enable/disable outlier detection
- `scale_method`: "standard", "minmax", "robust", or null

### Feature Engineering Options
- `create_domain_features`: Create housing-specific features
- `create_interactions`: Create feature interactions
- `feature_selection_method`: "k_best", "rfe", "importance_threshold"
- `k_features`: Number of features to select

### Model Training Options
- `perform_cv`: Enable cross-validation
- `tune_best_models`: Enable hyperparameter tuning
- `top_models_to_tune`: Number of top models to tune

## Pipeline Outputs

### Data Files
- `../data/processed_housing_data.csv` - Cleaned data
- `../data/engineered_housing_data.csv` - Final features

### Models
- `../models/best_model_[timestamp].pkl` - Trained model
- `../deployment/` - Complete deployment package

### Evaluation
- `../plots/` - Visualization plots
- `../reports/` - Evaluation reports
- `../logs/` - Execution logs

## Pipeline Components

### 1. Preprocessing (`../src/preprocessing.py`)
- Missing value imputation
- Duplicate removal
- Outlier detection and handling
- Feature scaling
- Data validation

### 2. Feature Engineering (`../src/feature_engineering.py`)
- Domain-specific feature creation
- Interaction features
- Polynomial features
- Feature selection
- Dimensionality reduction

### 3. Model Training (`../src/model_selection_training.py`)
- Multiple algorithm comparison
- Cross-validation
- Hyperparameter tuning
- Model persistence

### 4. Evaluation (`../src/evaluation.py`)
- Comprehensive metrics calculation
- Visualization generation
- Performance analysis
- Comparison reports

### 5. Deployment (`../src/deployment_monitoring.py`)
- Model packaging
- Monitoring setup
- Drift detection
- Performance tracking

## Usage Examples

### Run with Custom Configuration
```python
from main import HousingPricePipeline

# Load custom config
pipeline = HousingPricePipeline("custom_config.json")
success, results = pipeline.run_complete_pipeline()
```

### Run Individual Steps
```python
pipeline = HousingPricePipeline()

# Run only preprocessing
success = pipeline.run_preprocessing()

# Run only feature engineering (requires preprocessed data)
success = pipeline.run_feature_engineering()
```

### Access Results
```python
pipeline = HousingPricePipeline()
success, results = pipeline.run_complete_pipeline()

# Access specific step results
preprocessing_report = results["preprocessing"]["cleaning_report"]
best_model_name = results["model_training"]["best_model"]
evaluation_metrics = results["evaluation"]["evaluation_results"]["metrics"]
```

## Monitoring and Deployment

After successful pipeline execution:

1. **Model Deployment Package**: Found in `../deployment/`
   - `model.pkl` - Trained model
   - `metadata.json` - Model information
   - `deployment_config.json` - Deployment configuration

2. **Monitoring Setup**: Found in `../monitoring/`
   - Baseline statistics for drift detection
   - Performance tracking configuration
   - Monitoring reports

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the `app/` directory
2. **Missing Data**: Run `python ../data.py` to generate synthetic data
3. **Memory Issues**: Reduce `k_features` or disable polynomial features
4. **Long Execution**: Disable hyperparameter tuning for faster runs

### Error Logs
Check `../logs/pipeline.log` for detailed error information.

### Configuration Validation
The pipeline validates all file paths and parameters before execution.

## Performance Tips

1. **Faster Execution**:
   - Set `tune_best_models: false`
   - Reduce `top_models_to_tune`
   - Disable `create_polynomial`

2. **Better Results**:
   - Enable all feature engineering options
   - Increase `k_features`
   - Enable hyperparameter tuning

3. **Memory Optimization**:
   - Use `scale_method: "robust"`
   - Limit polynomial degree
   - Use feature selection

## Pipeline Metrics

The pipeline tracks and reports:
- Data quality metrics
- Feature engineering impact
- Model performance comparison
- Cross-validation results
- Test set evaluation
- Deployment readiness

## Next Steps

After pipeline completion:
1. Review evaluation plots in `../plots/`
2. Check model performance in `../reports/`
3. Deploy model using `../deployment/` package
4. Set up monitoring using `../monitoring/` configuration

For production deployment, consider:
- API wrapper for model serving
- Real-time monitoring dashboard
- Automated retraining triggers
- A/B testing framework
