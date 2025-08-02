"""
Feature Engineering Module
Handles feature creation, transformation, selection, and engineering for housing price prediction
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_selector = None
        self.pca = None
        self.poly_features = None
        self.label_encoders = {}
        self.feature_importance = {}
        self.engineered_features = []
        
    def load_processed_data(self, filepath):
        """Load preprocessed data"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Processed data loaded. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def create_domain_features(self, df):
        """Create domain-specific features for housing data"""
        df_features = df.copy()
        logger.info("Creating domain-specific features...")
        
        # Price per square foot
        if 'price' in df_features.columns and 'sqft_living' in df_features.columns:
            df_features['price_per_sqft'] = df_features['price'] / df_features['sqft_living']
            self.engineered_features.append('price_per_sqft')
        
        # Total square footage (living + lot)
        if 'sqft_living' in df_features.columns and 'sqft_lot' in df_features.columns:
            df_features['total_sqft'] = df_features['sqft_living'] + df_features['sqft_lot']
            self.engineered_features.append('total_sqft')
        
        # Rooms per bathroom ratio
        if 'bedrooms' in df_features.columns and 'bathrooms' in df_features.columns:
            df_features['bedrooms_to_bathrooms'] = df_features['bedrooms'] / (df_features['bathrooms'] + 1e-8)
            self.engineered_features.append('bedrooms_to_bathrooms')
        
        # Living space efficiency (living space / lot size)
        if 'sqft_living' in df_features.columns and 'sqft_lot' in df_features.columns:
            df_features['living_lot_ratio'] = df_features['sqft_living'] / (df_features['sqft_lot'] + 1e-8)
            self.engineered_features.append('living_lot_ratio')
        
        # Age categories
        if 'age' in df_features.columns:
            df_features['age_category'] = pd.cut(df_features['age'], 
                                               bins=[0, 5, 15, 30, 50, 100], 
                                               labels=['New', 'Recent', 'Mature', 'Old', 'Historic'])
            # Convert to numeric for modeling
            le = LabelEncoder()
            df_features['age_category_encoded'] = le.fit_transform(df_features['age_category'].astype(str))
            self.label_encoders['age_category'] = le
            self.engineered_features.append('age_category_encoded')
        
        # Quality score (combination of condition and grade)
        if 'condition' in df_features.columns and 'grade' in df_features.columns:
            df_features['quality_score'] = (df_features['condition'] * 0.3 + df_features['grade'] * 0.7)
            self.engineered_features.append('quality_score')
        
        # Size category based on living space
        if 'sqft_living' in df_features.columns:
            df_features['size_category'] = pd.cut(df_features['sqft_living'],
                                                bins=[0, 1000, 1500, 2500, 4000, 10000],
                                                labels=['Small', 'Medium', 'Large', 'XLarge', 'Mansion'])
            # Convert to numeric
            le = LabelEncoder()
            df_features['size_category_encoded'] = le.fit_transform(df_features['size_category'].astype(str))
            self.label_encoders['size_category'] = le
            self.engineered_features.append('size_category_encoded')
        
        # Luxury indicator (high grade + good condition)
        if 'condition' in df_features.columns and 'grade' in df_features.columns:
            df_features['is_luxury'] = ((df_features['grade'] >= 10) & (df_features['condition'] >= 4)).astype(int)
            self.engineered_features.append('is_luxury')
        
        # Space per room
        if 'sqft_living' in df_features.columns and 'bedrooms' in df_features.columns:
            df_features['sqft_per_bedroom'] = df_features['sqft_living'] / (df_features['bedrooms'] + 1e-8)
            self.engineered_features.append('sqft_per_bedroom')
        
        logger.info(f"Created {len(self.engineered_features)} domain-specific features")
        return df_features
    
    def create_interaction_features(self, df, max_interactions=5):
        """Create interaction features between important variables"""
        df_features = df.copy()
        logger.info("Creating interaction features...")
        
        # Key features that often interact in housing prices
        key_features = []
        potential_features = ['sqft_living', 'bedrooms', 'bathrooms', 'grade', 'condition', 'age']
        
        for feature in potential_features:
            if feature in df_features.columns:
                key_features.append(feature)
        
        interaction_count = 0
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                if interaction_count < max_interactions:
                    # Multiplicative interaction
                    interaction_name = f"{feat1}_x_{feat2}"
                    df_features[interaction_name] = df_features[feat1] * df_features[feat2]
                    self.engineered_features.append(interaction_name)
                    interaction_count += 1
        
        logger.info(f"Created {interaction_count} interaction features")
        return df_features
    
    def create_polynomial_features(self, df, degree=2, target_column='price'):
        """Create polynomial features for numeric columns"""
        logger.info(f"Creating polynomial features (degree={degree})...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        # Limit to most important features to avoid feature explosion
        important_cols = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        
        if len(important_cols) > 0:
            self.poly_features = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
            poly_array = self.poly_features.fit_transform(df[important_cols])
            
            # Create column names for polynomial features
            poly_feature_names = self.poly_features.get_feature_names_out(important_cols)
            poly_df = pd.DataFrame(poly_array, columns=poly_feature_names, index=df.index)
            
            # Remove original features (they're included in poly features)
            poly_df = poly_df.drop(columns=important_cols)
            
            # Combine with original dataframe
            df_poly = pd.concat([df, poly_df], axis=1)
            
            new_features = list(poly_df.columns)
            self.engineered_features.extend(new_features)
            
            logger.info(f"Created {len(new_features)} polynomial features")
            return df_poly
        
        logger.warning("No suitable columns found for polynomial features")
        return df
    
    def create_binned_features(self, df, columns=None):
        """Create binned versions of continuous features"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in columns if col != 'price']  # Exclude target
        
        df_binned = df.copy()
        logger.info("Creating binned features...")
        
        for col in columns:
            if col in df_binned.columns:
                # Create equal-width bins
                n_bins = min(5, df_binned[col].nunique())  # Max 5 bins
                if n_bins > 1:
                    binned_col = f"{col}_binned"
                    df_binned[binned_col] = pd.cut(df_binned[col], bins=n_bins, labels=False)
                    self.engineered_features.append(binned_col)
        
        logger.info(f"Created binned features for {len(columns)} columns")
        return df_binned
    
    def calculate_feature_importance(self, X, y, method='random_forest'):
        """Calculate feature importance using various methods"""
        logger.info(f"Calculating feature importance using {method}...")
        
        if method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importance_scores = rf.feature_importances_
            
        elif method == 'mutual_info':
            importance_scores = mutual_info_regression(X, y, random_state=42)
            
        elif method == 'f_regression':
            f_scores, _ = f_regression(X, y)
            importance_scores = f_scores
        
        # Create importance dictionary
        feature_importance = dict(zip(X.columns, importance_scores))
        self.feature_importance[method] = feature_importance
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Feature importance calculated using {method}")
        return sorted_features
    
    def select_features(self, X, y, method='k_best', k=15):
        """Select best features using various methods"""
        logger.info(f"Selecting features using {method} (k={k})...")
        
        if method == 'k_best':
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
        elif method == 'rfe':
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            self.feature_selector = RFE(estimator=rf, n_features_to_select=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
        elif method == 'importance_threshold':
            # Select features based on importance threshold
            importance_scores = self.calculate_feature_importance(X, y, 'random_forest')
            threshold = np.percentile([score for _, score in importance_scores], 70)  # Top 30%
            selected_features = [feat for feat, score in importance_scores if score >= threshold]
            X_selected = X[selected_features]
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        return X_selected, selected_features
    
    def apply_pca(self, X, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        logger.info(f"Applying PCA with {n_components} components...")
        
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        # Create column names for PCA components
        pca_columns = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA completed. Explained variance: {explained_variance:.3f}")
        
        return X_pca_df
    
    def feature_engineering_pipeline(self, filepath, target_column='price', 
                                   create_domain=True, create_interactions=True,
                                   create_polynomial=False, create_binned=True,
                                   feature_selection_method='k_best', k_features=15,
                                   apply_pca_flag=False, pca_components=0.95):
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
        # Load data
        df = self.load_processed_data(filepath)
        
        # Create domain-specific features
        if create_domain:
            df = self.create_domain_features(df)
        
        # Create interaction features
        if create_interactions:
            df = self.create_interaction_features(df, max_interactions=5)
        
        # Create polynomial features
        if create_polynomial:
            df = self.create_polynomial_features(df, degree=2, target_column=target_column)
        
        # Create binned features
        if create_binned:
            df = self.create_binned_features(df)
        
        # Separate features and target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Remove any non-numeric columns that might have been created
        X = X.select_dtypes(include=[np.number])
        
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        
        # Calculate feature importance
        importance_scores = self.calculate_feature_importance(X, y, 'random_forest')
        
        # Feature selection
        if feature_selection_method and k_features < len(X.columns):
            X_selected, selected_features = self.select_features(X, y, feature_selection_method, k_features)
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        # Apply PCA if requested
        if apply_pca_flag:
            X = self.apply_pca(X, pca_components)
        
        # Combine back with target
        final_df = pd.concat([X, y], axis=1)
        
        # Create feature engineering report
        fe_report = {
            'original_features': len(df.columns) - 1,  # Exclude target
            'engineered_features_created': len(self.engineered_features),
            'final_features': len(X.columns),
            'feature_selection_method': feature_selection_method,
            'selected_features': list(X.columns) if feature_selection_method else None,
            'top_features': [feat for feat, _ in importance_scores[:10]],
            'pca_applied': apply_pca_flag,
            'final_shape': final_df.shape
        }
        
        logger.info(f"Feature engineering completed. Final shape: {final_df.shape}")
        return final_df, X, y, fe_report
    
    def save_engineered_data(self, df, filepath):
        """Save engineered features to CSV"""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"Engineered data saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving engineered data: {str(e)}")
            raise

def main():
    """Example usage of the feature engineering module"""
    engineer = FeatureEngineer()
    
    # Define file paths
    input_file = "../data/processed_housing_data.csv"
    output_file = "../data/engineered_housing_data.csv"
    
    try:
        # Run feature engineering pipeline
        final_df, X, y, fe_report = engineer.feature_engineering_pipeline(
            filepath=input_file,
            target_column='price',
            create_domain=True,
            create_interactions=True,
            create_polynomial=False,  # Can be memory intensive
            create_binned=True,
            feature_selection_method='k_best',
            k_features=15,
            apply_pca_flag=False
        )
        
        # Save engineered data
        engineer.save_engineered_data(final_df, output_file)
        
        # Print summary
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Original features: {fe_report['original_features']}")
        print(f"Engineered features created: {fe_report['engineered_features_created']}")
        print(f"Final features: {fe_report['final_features']}")
        print(f"Final shape: {fe_report['final_shape']}")
        print(f"Feature selection method: {fe_report['feature_selection_method']}")
        
        if fe_report['top_features']:
            print(f"\nTop 10 most important features:")
            for i, feature in enumerate(fe_report['top_features'], 1):
                print(f"{i:2d}. {feature}")
        
        return final_df, fe_report
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
