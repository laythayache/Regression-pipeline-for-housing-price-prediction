"""
Data Preprocessing Module
Handles data cleaning, missing value imputation, duplicate removal, and basic data validation
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.original_shape = None
        self.cleaning_report = {}
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            df = pd.read_csv(filepath)
            self.original_shape = df.shape
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_data_quality(self, df):
        """Analyze data quality and return comprehensive report"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Missing values analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_percent, 2)
            }
        
        # Statistical outliers detection (using IQR method)
        outliers_report = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outliers_report[col] = {
                'count': len(outliers),
                'percentage': round((len(outliers) / len(df)) * 100, 2)
            }
        
        report['outliers'] = outliers_report
        
        logger.info(f"Data quality analysis completed")
        return report
    
    def handle_missing_values(self, df, strategy='mean', columns=None):
        """Handle missing values using various strategies"""
        if columns is None:
            columns = df.columns
        
        df_clean = df.copy()
        missing_before = df_clean.isnull().sum().sum()
        
        if strategy == 'drop_rows':
            df_clean = df_clean.dropna()
            
        elif strategy == 'drop_columns':
            # Drop columns with more than 50% missing values
            threshold = 0.5
            missing_percentages = df_clean.isnull().sum() / len(df_clean)
            cols_to_drop = missing_percentages[missing_percentages > threshold].index
            df_clean = df_clean.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns with >50% missing values: {list(cols_to_drop)}")
            
        elif strategy in ['mean', 'median', 'mode', 'constant']:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            
            # Handle numeric columns
            if len(numeric_cols) > 0:
                if strategy == 'mean':
                    self.imputer = SimpleImputer(strategy='mean')
                elif strategy == 'median':
                    self.imputer = SimpleImputer(strategy='median')
                elif strategy == 'constant':
                    self.imputer = SimpleImputer(strategy='constant', fill_value=0)
                
                df_clean[numeric_cols] = self.imputer.fit_transform(df_clean[numeric_cols])
            
            # Handle categorical columns with mode
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    mode_value = df_clean[col].mode()
                    if len(mode_value) > 0:
                        df_clean[col].fillna(mode_value[0], inplace=True)
                    else:
                        df_clean[col].fillna('Unknown', inplace=True)
                        
        elif strategy == 'knn':
            # Use KNN imputation for numeric columns only
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.imputer = KNNImputer(n_neighbors=5)
                df_clean[numeric_cols] = self.imputer.fit_transform(df_clean[numeric_cols])
        
        missing_after = df_clean.isnull().sum().sum()
        
        self.cleaning_report['missing_values'] = {
            'strategy': strategy,
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'missing_removed': int(missing_before - missing_after)
        }
        
        logger.info(f"Missing values handled using {strategy} strategy. "
                   f"Before: {missing_before}, After: {missing_after}")
        
        return df_clean
    
    def remove_duplicates(self, df, keep='first'):
        """Remove duplicate rows"""
        duplicates_before = df.duplicated().sum()
        df_clean = df.drop_duplicates(keep=keep)
        duplicates_removed = duplicates_before - df_clean.duplicated().sum()
        
        self.cleaning_report['duplicates'] = {
            'duplicates_before': int(duplicates_before),
            'duplicates_removed': int(duplicates_removed),
            'keep_strategy': keep
        }
        
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        return df_clean
    
    def handle_outliers(self, df, method='iqr', columns=None, factor=1.5):
        """Handle outliers using various methods"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_clean = df.copy()
        outliers_removed = 0
        
        if method == 'iqr':
            for col in columns:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                    outliers_count = outliers_mask.sum()
                    outliers_removed += outliers_count
                    
                    # Remove outliers
                    df_clean = df_clean[~outliers_mask]
                    
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                if col in df_clean.columns:
                    z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                    outliers_mask = z_scores > 3
                    outliers_count = outliers_mask.sum()
                    outliers_removed += outliers_count
                    
                    # Remove outliers
                    df_clean = df_clean[z_scores <= 3]
        
        self.cleaning_report['outliers'] = {
            'method': method,
            'outliers_removed': int(outliers_removed),
            'factor': factor
        }
        
        logger.info(f"Removed {outliers_removed} outliers using {method} method")
        return df_clean
    
    def validate_data_types(self, df, expected_types=None):
        """Validate and convert data types"""
        df_clean = df.copy()
        
        # Auto-detect and convert obvious numeric columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except:
                    pass
        
        # Apply expected types if provided
        if expected_types:
            for col, dtype in expected_types.items():
                if col in df_clean.columns:
                    try:
                        df_clean[col] = df_clean[col].astype(dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
        
        logger.info("Data type validation completed")
        return df_clean
    
    def scale_features(self, df, method='standard', exclude_columns=None):
        """Scale numerical features"""
        if exclude_columns is None:
            exclude_columns = []
        
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for scaling")
            return df_scaled
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_scaled[numeric_cols] = self.scaler.fit_transform(df_scaled[numeric_cols])
        
        logger.info(f"Features scaled using {method} method")
        return df_scaled
    
    def preprocess_pipeline(self, filepath, missing_strategy='mean', 
                          remove_duplicates=True, handle_outliers=True, 
                          outlier_method='iqr', scale_method=None, 
                          target_column='price'):
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        df = self.load_data(filepath)
        
        # Analyze data quality
        quality_report = self.analyze_data_quality(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy=missing_strategy)
        
        # Remove duplicates
        if remove_duplicates:
            df = self.remove_duplicates(df)
        
        # Handle outliers (exclude target column)
        if handle_outliers:
            outlier_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col != target_column]
            df = self.handle_outliers(df, method=outlier_method, columns=outlier_columns)
        
        # Validate data types
        df = self.validate_data_types(df)
        
        # Scale features (exclude target column)
        if scale_method:
            df = self.scale_features(df, method=scale_method, exclude_columns=[target_column])
        
        # Final quality check
        final_quality = self.analyze_data_quality(df)
        
        self.cleaning_report['final_shape'] = df.shape
        self.cleaning_report['original_shape'] = self.original_shape
        self.cleaning_report['rows_removed'] = self.original_shape[0] - df.shape[0]
        
        logger.info(f"Preprocessing completed. Final shape: {df.shape}")
        return df, quality_report, final_quality, self.cleaning_report
    
    def save_processed_data(self, df, filepath):
        """Save processed data to CSV"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"Processed data saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

def main():
    """Example usage of the preprocessing module"""
    preprocessor = DataPreprocessor()
    
    # Define file paths
    input_file = "../data/housing_data.csv"
    output_file = "../data/processed_housing_data.csv"
    
    try:
        # Run preprocessing pipeline
        processed_df, initial_quality, final_quality, cleaning_report = preprocessor.preprocess_pipeline(
            filepath=input_file,
            missing_strategy='mean',
            remove_duplicates=True,
            handle_outliers=True,
            outlier_method='iqr',
            scale_method='standard',
            target_column='price'
        )
        
        # Save processed data
        preprocessor.save_processed_data(processed_df, output_file)
        
        # Print summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Original shape: {cleaning_report['original_shape']}")
        print(f"Final shape: {cleaning_report['final_shape']}")
        print(f"Rows removed: {cleaning_report['rows_removed']}")
        print(f"Missing values removed: {cleaning_report['missing_values']['missing_removed']}")
        print(f"Duplicates removed: {cleaning_report['duplicates']['duplicates_removed']}")
        print(f"Outliers removed: {cleaning_report['outliers']['outliers_removed']}")
        
        return processed_df, cleaning_report
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()