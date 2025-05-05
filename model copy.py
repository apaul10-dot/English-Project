 # Enhanced Python Implementation for Agricultural Food Waste Analysis
# ==================================================================

# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from bayes_opt import BayesianOptimization
import optuna
import xgboost as xgb
import lightgbm as lgb
import os
import json
import joblib
import warnings
import time
from functools import partial
from itertools import product
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 1. Data Acquisition and Integration
# ==================================

def download_and_prepare_data():
    """
    Load the three datasets: FAO.csv, Crop_Recommendation.csv, and crop_yield.csv
    """
    # Path setup
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1.1 Load FAO Food Loss and Waste Database
    print("Loading FAO Food Loss and Waste Database...")
    try:
        fao_data = pd.read_csv(f"{data_dir}/FAO.csv")
        print(f"FAO data loaded with {fao_data.shape[0]} rows and {fao_data.shape[1]} columns")
    except FileNotFoundError:
        print("FAO.csv not found in data directory")
        fao_data = None
    
    # 1.2 Load Crop Recommendation Dataset
    print("Loading Crop Recommendation Dataset...")
    try:
        crop_rec_data = pd.read_csv(f"{data_dir}/Crop_Recommendation.csv")
        print(f"Crop recommendation data loaded with {crop_rec_data.shape[0]} rows and {crop_rec_data.shape[1]} columns")
    except FileNotFoundError:
        print("Crop_Recommendation.csv not found in data directory")
        crop_rec_data = None
        
    # 1.3 Load Agricultural Crop Yield Dataset
    print("Loading Agricultural Crop Yield Dataset...")
    try:
        crop_yield_data = pd.read_csv(f"{data_dir}/crop_yield.csv")
        print(f"Crop yield data loaded with {crop_yield_data.shape[0]} rows and {crop_yield_data.shape[1]} columns")
    except FileNotFoundError:
        print("crop_yield.csv not found in data directory")
        crop_yield_data = None
    
    # Summary of loaded data
    print("\nData Loading Summary:")
    print(f"FAO Data: {'Loaded' if fao_data is not None else 'Not Found'}")
    print(f"Crop Recommendation Data: {'Loaded' if crop_rec_data is not None else 'Not Found'}")
    print(f"Crop Yield Data: {'Loaded' if crop_yield_data is not None else 'Not Found'}")
    
    return fao_data, crop_rec_data, crop_yield_data

# 2. Initial Data Cleaning
# ======================

def clean_data(fao_data, crop_rec_data, crop_yield_data):
    """
    Perform initial data cleaning on all datasets with enhanced methods.
    """
    print("\nPerforming enhanced data cleaning...")
    
    # 2.1 Clean FAO data
    if fao_data is not None:
        print("Cleaning FAO data...")
        # Filter for India only
        fao_data_clean = fao_data[fao_data['country'] == 'India'].copy()
        print(f"Filtered FAO data for India: {fao_data_clean.shape[0]} rows")
        
        # Convert loss_percentage to numeric
        fao_data_clean['loss_percentage'] = pd.to_numeric(fao_data_clean['loss_percentage'], errors='coerce')
        
        # Standardize crop names
        fao_data_clean['commodity'] = fao_data_clean['commodity'].str.strip().str.capitalize()
        
        # Enhanced cleaning: detect and handle outliers in loss_percentage using IQR
        Q1 = fao_data_clean['loss_percentage'].quantile(0.25)
        Q3 = fao_data_clean['loss_percentage'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers rather than removing them
        fao_data_clean['loss_percentage'] = fao_data_clean['loss_percentage'].clip(lower_bound, upper_bound)
        
        # Fill missing loss percentages with group medians
        if 'food_supply_stage' in fao_data_clean.columns:
            stage_medians = fao_data_clean.groupby('food_supply_stage')['loss_percentage'].median()
            for stage in stage_medians.index:
                mask = (fao_data_clean['food_supply_stage'] == stage) & (fao_data_clean['loss_percentage'].isna())
                fao_data_clean.loc[mask, 'loss_percentage'] = stage_medians[stage]
        
        # Fill any remaining NaNs with overall median
        fao_data_clean['loss_percentage'].fillna(fao_data_clean['loss_percentage'].median(), inplace=True)
    else:
        fao_data_clean = None
    
    # 2.2 Clean Crop Recommendation data
    if crop_rec_data is not None:
        print("Cleaning crop recommendation data...")
        crop_rec_clean = crop_rec_data.copy()
        
        # Standardize column names
        crop_rec_clean.columns = [col.lower() for col in crop_rec_clean.columns]
        
        # Rename columns to match expected format
        if 'crop' not in crop_rec_clean.columns and 'label' in crop_rec_clean.columns:
            crop_rec_clean.rename(columns={'label': 'crop'}, inplace=True)
        
        # Check if pH column needs renaming
        if 'ph_value' in crop_rec_clean.columns:
            crop_rec_clean.rename(columns={'ph_value': 'ph'}, inplace=True)
        
        # Standardize crop names
        crop_rec_clean['crop'] = crop_rec_clean['crop'].str.strip().str.capitalize()
        
        # Enhanced cleaning: Use IQR method for outlier detection
        features = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        valid_features = [col for col in features if col in crop_rec_clean.columns]
        
        for column in valid_features:
            Q1 = crop_rec_clean[column].quantile(0.25)
            Q3 = crop_rec_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers rather than removing them
            crop_rec_clean[column] = crop_rec_clean[column].clip(lower_bound, upper_bound)
        
        # Check for and handle missing values
        if crop_rec_clean.isnull().sum().any():
            # Use KNN imputer for better handling of relationships between features
            imputer = KNNImputer(n_neighbors=5)
            crop_rec_clean[valid_features] = imputer.fit_transform(crop_rec_clean[valid_features])
    else:
        crop_rec_clean = None
    
    # 2.3 Clean Crop Yield data
    if crop_yield_data is not None:
        print("Cleaning crop yield data...")
        crop_yield_clean = crop_yield_data.copy()
        
        # Standardize column names
        crop_yield_clean.columns = [col.lower() for col in crop_yield_clean.columns]
        
        # Standardize crop names
        crop_yield_clean['crop'] = crop_yield_clean['crop'].str.strip().str.capitalize()
        
        # Convert crop_year to numeric if it's not already
        if 'crop_year' in crop_yield_clean.columns:
            crop_yield_clean['crop_year'] = pd.to_numeric(crop_yield_clean['crop_year'], errors='coerce')
        
        # Enhanced cleaning for yield data
        # Handle outliers using IQR method per crop type
        crops = crop_yield_clean['crop'].unique()
        
        for crop in crops:
            crop_df = crop_yield_clean[crop_yield_clean['crop'] == crop]
            
            if 'yield' in crop_df.columns and len(crop_df) > 10:  # Only process if enough data
                Q1 = crop_df['yield'].quantile(0.25)
                Q3 = crop_df['yield'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                mask = crop_yield_clean['crop'] == crop
                crop_yield_clean.loc[mask, 'yield'] = crop_yield_clean.loc[mask, 'yield'].clip(lower_bound, upper_bound)
        
        # Handle missing values in production/yield data
        numeric_cols = crop_yield_clean.select_dtypes(include=[np.number]).columns
        if crop_yield_clean[numeric_cols].isnull().sum().any():
            # Group by crop and region for imputation
            group_cols = [col for col in ['crop', 'state', 'season'] if col in crop_yield_clean.columns]
            
            if group_cols:
                # Fill NAs with group means
                for col in numeric_cols:
                    if crop_yield_clean[col].isnull().any():
                        group_means = crop_yield_clean.groupby(group_cols)[col].transform('mean')
                        crop_yield_clean[col].fillna(group_means, inplace=True)
            
            # Fill any remaining NAs with column medians
            for col in numeric_cols:
                crop_yield_clean[col].fillna(crop_yield_clean[col].median(), inplace=True)
    else:
        crop_yield_clean = None
    
    return fao_data_clean, crop_rec_clean, crop_yield_clean

# 3. Data Transformation and Normalization
# ======================================

def transform_and_normalize_data(fao_data, crop_rec_data, crop_yield_data):
    """
    Apply enhanced transformations and normalization to prepare data for modeling.
    """
    print("\nPerforming advanced data transformation and normalization...")
    
    # 3.1 Transform and normalize FAO data
    if fao_data is not None:
        print("Transforming FAO data...")
        fao_transformed = fao_data.copy()
        
        # Normalize loss_percentage using robust scaler (resistant to outliers)
        if 'loss_percentage' in fao_transformed.columns:
            scaler = RobustScaler()
            fao_transformed['loss_percentage_scaled'] = scaler.fit_transform(
                fao_transformed[['loss_percentage']])
        
        # One-hot encode food_supply_stage and activity with handling for rare categories
        categorical_cols = [col for col in ['food_supply_stage', 'activity'] if col in fao_transformed.columns]
        
        for col in categorical_cols:
            # Only encode if the column has values
            if not fao_transformed[col].isna().all():
                # Handle rare categories
                value_counts = fao_transformed[col].value_counts()
                # Categories appearing less than 3 times are grouped as 'Other'
                rare_categories = value_counts[value_counts < 3].index.tolist()
                
                if rare_categories:
                    fao_transformed[col] = fao_transformed[col].apply(
                        lambda x: 'Other' if x in rare_categories else x)
                
                # One-hot encode
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded_data = encoder.fit_transform(fao_transformed[[col]].fillna('Missing'))
                encoded_df = pd.DataFrame(
                    encoded_data, 
                    columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                )
                fao_transformed = pd.concat([fao_transformed.reset_index(drop=True), 
                                           encoded_df.reset_index(drop=True)], axis=1)
    else:
        fao_transformed = None
    
    # 3.2 Transform and normalize crop recommendation data
    if crop_rec_data is not None:
        print("Transforming crop recommendation data...")
        crop_rec_transformed = crop_rec_data.copy()
        
        # Standardize numerical features
        numeric_cols = [col for col in ['nitrogen', 'phosphorus', 'potassium', 
                                       'temperature', 'humidity', 'ph', 'rainfall'] 
                        if col in crop_rec_transformed.columns]
        
        if numeric_cols:
            # Test for normality and apply appropriate transformations
            for col in numeric_cols:
                # Check skewness
                skewness = crop_rec_transformed[col].skew()
                
                # Apply appropriate transformation for skewed data
                if abs(skewness) > 1:  # Moderately skewed
                    if skewness > 0:  # Right-skewed
                        # For positive data, try log transform
                        if (crop_rec_transformed[col] > 0).all():
                            crop_rec_transformed[f'{col}_log'] = np.log1p(crop_rec_transformed[col])
                    else:  # Left-skewed
                        # Consider power transform
                        try:
                            crop_rec_transformed[f'{col}_transformed'], _ = yeojohnson(crop_rec_transformed[col])
                        except:
                            pass  # Skip if transformation fails
            
            # Standard scaling for all numeric features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(crop_rec_transformed[numeric_cols])
            
            for i, col in enumerate(numeric_cols):
                crop_rec_transformed[f'{col}_scaled'] = scaled_features[:, i]
    else:
        crop_rec_transformed = None
    
    # 3.3 Transform and normalize crop yield data
    if crop_yield_data is not None:
        print("Transforming crop yield data...")
        crop_yield_transformed = crop_yield_data.copy()
        
        # Handle 'yield' variable - check if transformation needed
        if 'yield' in crop_yield_transformed.columns:
            skewness = crop_yield_transformed['yield'].skew()
            
            if abs(skewness) > 1:  # If moderately skewed
                try:
                    crop_yield_transformed['yield_transformed'], _ = yeojohnson(crop_yield_transformed['yield'])
                except:
                    # Fall back to simple log transform
                    if (crop_yield_transformed['yield'] > 0).all():
                        crop_yield_transformed['yield_transformed'] = np.log1p(crop_yield_transformed['yield'])
                    else:
                        # Add offset for non-positive values
                        min_val = crop_yield_transformed['yield'].min()
                        if min_val <= 0:
                            offset = abs(min_val) + 1
                            crop_yield_transformed['yield_transformed'] = np.log1p(
                                crop_yield_transformed['yield'] + offset)
        
        # Handle 'production' variable
        if 'production' in crop_yield_transformed.columns:
            if (crop_yield_transformed['production'] > 0).all():
                crop_yield_transformed['production_transformed'] = np.log1p(
                    crop_yield_transformed['production'])
            else:
                # Add offset for non-positive values
                min_val = crop_yield_transformed['production'].min()
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    crop_yield_transformed['production_transformed'] = np.log1p(
                        crop_yield_transformed['production'] + offset)
        
        # One-hot encode categorical variables with handling for rare categories
        categorical_cols = [col for col in ['state', 'season', 'crop'] 
                           if col in crop_yield_transformed.columns]
        
        for col in categorical_cols:
            value_counts = crop_yield_transformed[col].value_counts()
            # Categories appearing less than 3 times are grouped as 'Other'
            rare_categories = value_counts[value_counts < 3].index.tolist()
            
            if rare_categories:
                crop_yield_transformed[col] = crop_yield_transformed[col].apply(
                    lambda x: 'Other' if x in rare_categories else x)
            
            # Skip 'crop' for one-hot encoding if we want to keep it as a feature
            if col != 'crop':
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded_data = encoder.fit_transform(crop_yield_transformed[[col]].fillna('Missing'))
                encoded_df = pd.DataFrame(
                    encoded_data, 
                    columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                )
                crop_yield_transformed = pd.concat([crop_yield_transformed.reset_index(drop=True), 
                                                 encoded_df.reset_index(drop=True)], axis=1)
    else:
        crop_yield_transformed = None
    
    return fao_transformed, crop_rec_transformed, crop_yield_transformed

# 4. Dataset Integration
# ====================

def integrate_datasets(fao_data, crop_rec_data, crop_yield_data):
    """
    Integrate the cleaned and transformed datasets with enhanced methods.
    """
    print("\nPerforming advanced dataset integration...")
    
    # Check if we have the necessary datasets
    if crop_yield_data is None:
        print("Crop yield data is required for integration. Aborting integration.")
        return None
    
    # Start with crop yield data as base
    integrated_data = crop_yield_data.copy()
    
    # 4.1 Create crop-wise mapping of optimal conditions from crop recommendation data
    if crop_rec_data is not None:
        print("Integrating crop recommendation data...")
        
        # Define aggregation functions for different columns
        agg_funcs = {}
        for col in crop_rec_data.columns:
            if col != 'crop':
                agg_funcs[col] = ['mean', 'median', 'std']
        
        # Aggregate crop recommendations by crop with multiple statistics
        crop_optimal_conditions = crop_rec_data.groupby('crop').agg(agg_funcs)
        # Flatten the column multi-index
        crop_optimal_conditions.columns = ['_'.join(col).strip() for col in crop_optimal_conditions.columns.values]
        crop_optimal_conditions.reset_index(inplace=True)
        
        # Merge with integrated data
        print(f"Merging yield data ({len(integrated_data)} rows) with crop recommendations ({len(crop_optimal_conditions)} rows)")
        integrated_data = integrated_data.merge(
            crop_optimal_conditions, 
            on='crop', 
            how='left'
        )
    else:
        print("No crop recommendation data available for integration")
    
    # 4.2 Add food waste percentages from FAO data
    if fao_data is not None:
        print("Integrating FAO food waste data...")
        
        # Create comprehensive waste metrics by crop and stage
        if 'food_supply_stage' in fao_data.columns:
            # Get multiple statistics for each crop-stage combination
            crop_waste_stats = fao_data.groupby(['commodity', 'food_supply_stage'])['loss_percentage'].agg([
                'mean', 'median', 'min', 'max', 'std'
            ]).reset_index()
            
            # Rename for clarity
            crop_waste_stats.rename(columns={
                'commodity': 'crop',
                'mean': 'waste_mean',
                'median': 'waste_median', 
                'min': 'waste_min',
                'max': 'waste_max',
                'std': 'waste_std'
            }, inplace=True)
            
            # Pivot to have stages as columns
            crop_waste_pivot = crop_waste_stats.pivot_table(
                index='crop', 
                columns='food_supply_stage',
                values=['waste_mean', 'waste_median', 'waste_min', 'waste_max', 'waste_std']
            )
            
            # Flatten the hierarchical column index
            crop_waste_pivot.columns = [f"{col[0]}_{col[1].replace(' ', '_').lower()}" 
                                       for col in crop_waste_pivot.columns]
            crop_waste_pivot.reset_index(inplace=True)
            
            # Focus on post-harvest waste for simplicity in merging
            post_harvest_cols = [col for col in crop_waste_pivot.columns 
                               if 'post' in col.lower() or 'harvest' in col.lower() or 'storage' in col.lower()]
            
            if post_harvest_cols and 'crop' in crop_waste_pivot.columns:
                post_harvest_waste = crop_waste_pivot[['crop'] + post_harvest_cols].copy()
                
                # Add an overall post-harvest waste column
                mean_cols = [col for col in post_harvest_cols if 'mean' in col]
                if mean_cols:
                    post_harvest_waste['post_harvest_waste_pct'] = post_harvest_waste[mean_cols].mean(axis=1)
                
                # Merge with integrated data
                print(f"Merging with FAO post-harvest waste data ({len(post_harvest_waste)} crops)")
                integrated_data = integrated_data.merge(
                    post_harvest_waste,
                    on='crop',
                    how='left'
                )
            else:
                # If no specific post-harvest data, use average waste per crop
                crop_waste_avg = fao_data.groupby('commodity')['loss_percentage'].agg([
                    'mean', 'median', 'min', 'max', 'std'
                ]).reset_index()
                crop_waste_avg.rename(columns={
                    'commodity': 'crop', 
                    'mean': 'post_harvest_waste_pct',
                    'median': 'waste_median',
                    'min': 'waste_min',
                    'max': 'waste_max',
                    'std': 'waste_std'
                }, inplace=True)
                
                integrated_data = integrated_data.merge(
                    crop_waste_avg,
                    on='crop',
                    how='left'
                )
        else:
            # Simple statistics by commodity
            crop_waste_stats = fao_data.groupby('commodity')['loss_percentage'].agg([
                'mean', 'median', 'min', 'max', 'std'
            ]).reset_index()
            crop_waste_stats.rename(columns={
                'commodity': 'crop', 
                'mean': 'post_harvest_waste_pct',
                'median': 'waste_median',
                'min': 'waste_min',
                'max': 'waste_max',
                'std': 'waste_std'
            }, inplace=True)
            
            integrated_data = integrated_data.merge(
                crop_waste_stats,
                on='crop',
                how='left'
            )
    else:
        print("No FAO data available for integration")
    
    # 4.3 Enhanced imputation for missing values in the integrated dataset
    print("Handling missing values in integrated dataset...")
    
    # First try to impute missing waste percentages using multiple regression
    if 'post_harvest_waste_pct' in integrated_data.columns and integrated_data['post_harvest_waste_pct'].isna().any():
        print(f"Imputing missing waste percentages for {integrated_data['post_harvest_waste_pct'].isna().sum()} rows")
        
        # Try multiple imputation models
        waste_model_data = integrated_data.dropna(subset=['post_harvest_waste_pct']).copy()
        
        # Select potential features that might influence waste
        potential_features = [
            'yield', 'temperature_mean', 'humidity_mean', 'rainfall_mean', 
            'ph_mean', 'nitrogen_mean', 'phosphorus_mean', 'potassium_mean',
            'crop_year'
        ]
        
        # Use only available features
        waste_features = [col for col in potential_features if col in waste_model_data.columns]
        
        # Drop rows with NaN in feature columns
        waste_model_data = waste_model_data.dropna(subset=waste_features)
        
        if len(waste_features) > 0 and len(waste_model_data) > 10:
            print(f"Training waste imputation model with {len(waste_model_data)} rows and {len(waste_features)} features")
            
            # Try different models and pick the best one
            models = {
                'Linear': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED)
            }
            
            best_model = None
            best_score = float('-inf')
            
            for name, model in models.items():
                try:
                    # 5-fold cross-validation
                    scores = cross_val_score(
                        model, waste_model_data[waste_features], 
                        waste_model_data['post_harvest_waste_pct'],
                        cv=5, scoring='r2'
                    )
                    mean_score = np.mean(scores)
                    
                    if mean_score > best_score and mean_score > 0:  # Only consider models with positive R2
                        best_score = mean_score
                        best_model = model
                        print(f"  {name} CV R² = {mean_score:.4f}")
                except:
                    print(f"  {name} failed in cross-validation")
            
            if best_model is not None:
                # Train final model on all data
                best_model.fit(waste_model_data[waste_features], waste_model_data['post_harvest_waste_pct'])
                
                # Predict missing values
                missing_waste_mask = integrated_data['post_harvest_waste_pct'].isna()
                missing_waste_data = integrated_data.loc[missing_waste_mask, waste_features]
                
                # Drop rows with NaN in feature columns
                valid_rows = ~missing_waste_data.isna().any(axis=1)
                if valid_rows.any():
                    predictions = best_model.predict(missing_waste_data.loc[valid_rows])
                    
                    # Ensure predictions are within reasonable bounds
                    min_waste = waste_model_data['post_harvest_waste_pct'].min()
                    max_waste = waste_model_data['post_harvest_waste_pct'].max()
                    predictions = np.clip(predictions, min_waste, max_waste)
                    
                    # Update only rows we could predict
                    rows_to_update = missing_waste_mask.index[missing_waste_mask][valid_rows]
                    integrated_data.loc[rows_to_update, 'post_harvest_waste_pct'] = predictions
                
                print(f"Imputed {sum(valid_rows)} missing waste percentage values with {best_model.__class__.__name__}")
    
    # 4.4 Advanced imputation for remaining missing values
    # KNN imputation for groups of related variables
    # Group columns by type
    columns = integrated_data.columns
    environmental_cols = [col for col in columns if any(x in col for x in ['temperature', 'humidity', 'rainfall', 'ph'])]
    nutrient_cols = [col for col in columns if any(x in col for x in ['nitrogen', 'phosphorus', 'potassium'])]
    production_cols = [col for col in columns if any(x in col for x in ['yield', 'production', 'area'])]
    waste_cols = [col for col in columns if 'waste' in col]
    
    # Apply KNN imputation within each group
    for col_group in [environmental_cols, nutrient_cols, production_cols, waste_cols]:
        if col_group and any(integrated_data[col_group].isna().any()):
            # Use KNN imputer for within-group relationships
            imputer = KNNImputer(n_neighbors=3)
            imputed_vals = imputer.fit_transform(integrated_data[col_group])
            
            # Update the dataframe
            for i, col in enumerate(col_group):
                integrated_data[col] = imputed_vals[:, i]
    
    # Fill any remaining missing values with column medians
    numeric_columns = integrated_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if integrated_data[col].isna().any():
            integrated_data[col].fillna(integrated_data[col].median(), inplace=True)
    
    print(f"Final integrated dataset: {integrated_data.shape[0]} rows, {integrated_data.shape[1]} columns")
    return integrated_data

# 5. Feature Engineering and Selection
# ==================================

def engineer_and_select_features(integrated_data):
    """
    Engineer new features and select the most relevant ones for modeling
    using advanced feature selection techniques.
    """
    print("\nPerforming advanced feature engineering and selection...")
    
    if integrated_data is None or len(integrated_data) == 0:
        print("No data available for feature engineering.")
        return None, None
    
    # Create a working copy
    data = integrated_data.copy()
    
    # 5.1 Advanced Feature Engineering
    print("Engineering new features...")
    
    # Soil quality and nutrient balance features
    if all(col in data.columns for col in ['nitrogen_mean', 'phosphorus_mean', 'potassium_mean']):
        # N-P-K ratio (normalized)
        npk_sum = data['nitrogen_mean'] + data['phosphorus_mean'] + data['potassium_mean']
        data['n_ratio'] = data['nitrogen_mean'] / npk_sum
        data['p_ratio'] = data['phosphorus_mean'] / npk_sum
        data['k_ratio'] = data['potassium_mean'] / npk_sum
        
        # NPK balance score - how close the ratio is to an ideal balance
        # Assume balanced NPK might be around 1:1:1 for general purposes
        data['npk_balance'] = 1 - (
            abs(data['n_ratio'] - 1/3) + 
            abs(data['p_ratio'] - 1/3) + 
            abs(data['k_ratio'] - 1/3)
        )
        
        # Overall soil fertility score
        data['soil_fertility'] = (
            data['nitrogen_mean'] / data['nitrogen_mean'].max() +
            data['phosphorus_mean'] / data['phosphorus_mean'].max() +
            data['potassium_mean'] / data['potassium_mean'].max()
        ) / 3
    
    # Environmental suitability features
    env_features = ['temperature_mean', 'humidity_mean', 'rainfall_mean', 'ph_mean']
    available_env = [col for col in env_features if col in data.columns]
    
    if len(available_env) >= 2:
        # Temperature-humidity interaction (heat index proxy)
        if all(col in data.columns for col in ['temperature_mean', 'humidity_mean']):
            data['temp_humidity_index'] = data['temperature_mean'] * data['humidity_mean'] / 100
        
        # Aridity index (rainfall to temperature ratio)
        if all(col in data.columns for col in ['rainfall_mean', 'temperature_mean']):
            data['aridity_index'] = data['rainfall_mean'] / (data['temperature_mean'] + 1)  # +1 to avoid division by zero
        
        # pH optimality (most crops prefer pH 5.5-7.5)
        if 'ph_mean' in data.columns:
            data['ph_optimality'] = 1 - abs((data['ph_mean'] - 6.5) / 3)
            data['ph_optimality'] = data['ph_optimality'].clip(0, 1)  # Bound between 0 and 1
    
    # Crop-specific features
    if 'crop' in data.columns:
        # Create crop type categories
        grain_crops = ['Rice', 'Wheat', 'Maize', 'Barley', 'Sorghum', 'Millet']
        pulse_crops = ['Gram', 'Chickpea', 'Lentil', 'Pigeon Pea', 'Black gram', 'Green gram']
        oil_crops = ['Groundnut', 'Sesamum', 'Sunflower', 'Safflower', 'Castor', 'Linseed', 'Rapeseed', 'Mustard']
        cash_crops = ['Sugarcane', 'Cotton', 'Jute', 'Tobacco']
        fruit_veg = ['Potato', 'Onion', 'Tomato', 'Cauliflower', 'Cabbage', 'Brinjal', 'Ladies Finger', 'Peas']
        
        # Function to categorize crops
        def categorize_crop(crop):
            if crop in grain_crops:
                return 'Grain'
            elif crop in pulse_crops:
                return 'Pulse'
            elif crop in oil_crops:
                return 'Oil'
            elif crop in cash_crops:
                return 'Cash'
            elif crop in fruit_veg:
                return 'Fruit/Vegetable'
            else:
                return 'Other'
        
        data['crop_category'] = data['crop'].apply(categorize_crop)
        
        # One-hot encode crop category
        crop_cat_dummies = pd.get_dummies(data['crop_category'], prefix='crop_cat', drop_first=True)
        data = pd.concat([data, crop_cat_dummies], axis=1)
    
    # Season-based features
    if 'season' in data.columns:
        # Identify main growing seasons in India
        data['is_kharif'] = data['season'].str.contains('Kharif|kharif|Monsoon|monsoon').astype(int)
        data['is_rabi'] = data['season'].str.contains('Rabi|rabi|Winter|winter').astype(int)
        data['is_summer'] = data['season'].str.contains('Summer|summer|Zaid|zaid').astype(int)
    
    # Time-based features
    if 'crop_year' in data.columns:
        # Create decade feature
        data['decade'] = (data['crop_year'] // 10) * 10
        
        # Create trend features
        min_year = data['crop_year'].min()
        data['years_since_min'] = data['crop_year'] - min_year
    
    # Waste-Yield interaction features
    if all(col in data.columns for col in ['yield', 'post_harvest_waste_pct']):
        # Calculate effective yield (after waste)
        data['effective_yield'] = data['yield'] * (1 - data['post_harvest_waste_pct']/100)
        
        # Calculate waste amount per area
        data['waste_per_area'] = data['yield'] * (data['post_harvest_waste_pct']/100)
    
    # 5.2 Advanced Feature Selection
    print("Selecting optimal features...")
    
    # Prepare data for feature selection
    # Remove ID columns, target columns, and non-numeric columns for correlation analysis
    exclude_patterns = ['id', 'code', 'name', 'year', 'index']
    exclude_cols = [col for col in data.columns if any(pattern in col.lower() for pattern in exclude_patterns)]
    
    # Identify target columns
    target_cols = ['yield', 'post_harvest_waste_pct', 'effective_yield']
    available_targets = [col for col in target_cols if col in data.columns]
    
    if not available_targets:
        print("No target variables found. Cannot perform feature selection.")
        return data, data.columns.tolist()
    
    # Remove target columns from feature set
    exclude_cols += available_targets
    
    # Select numeric columns for analysis, excluding specified ones
    X_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
              if col not in exclude_cols]
    
    # Remove columns with too many missing values
    X_cols = [col for col in X_cols if data[col].isna().mean() < 0.3]
    
    # Check if we have enough data for feature selection
    if len(X_cols) < 3:
        print("Not enough features available for selection.")
        return data, X_cols
    
    # Create feature matrix
    X = data[X_cols].copy()
    
    # 5.3 Check for multicollinearity
    print("Checking for multicollinearity...")
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    high_corr_pairs = []
    
    # Find feature pairs with high correlation
    high_corr = corr_matrix.where(mask).stack()
    high_corr = high_corr[high_corr > 0.85]  # Threshold for high correlation
    
    if not high_corr.empty:
        high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]) 
                          for i, j in zip(*np.where(corr_matrix.values > 0.85)) if i != j]
        
        print(f"Found {len(high_corr_pairs)} highly correlated feature pairs:")
        for feat1, feat2, corr in high_corr_pairs[:5]:  # Show just first 5 for brevity
            print(f"  {feat1} and {feat2}: {corr:.4f}")
        
        # Remove one feature from each highly correlated pair
        # We'll keep track of features to drop
        drop_features = set()
        
        for feat1, feat2, _ in high_corr_pairs:
            # If both features are still in consideration
            if feat1 not in drop_features and feat2 not in drop_features:
                # Look at correlation with targets to decide which to keep
                corr_with_targets = {}
                
                for feat in [feat1, feat2]:
                    # Calculate mean absolute correlation with available targets
                    target_corrs = []
                    for target in available_targets:
                        if target in data.columns:
                            target_corrs.append(abs(np.corrcoef(data[feat], data[target])[0, 1]))
                    
                    if target_corrs:
                        corr_with_targets[feat] = np.mean(target_corrs)
                    else:
                        corr_with_targets[feat] = 0
                
                # Drop the feature with lower correlation to targets
                if corr_with_targets[feat1] >= corr_with_targets[feat2]:
                    drop_features.add(feat2)
                else:
                    drop_features.add(feat1)
        
        print(f"Removing {len(drop_features)} features due to high collinearity")
        
        # Remove the identified features
        X_cols = [col for col in X_cols if col not in drop_features]
        X = data[X_cols].copy()
    
    # 5.4 Advanced feature selection using a combination of methods
    print("Performing multi-method feature selection...")
    
    selected_features = {}
    
    # For each target, perform feature selection
    for target in available_targets:
        if target in data.columns:
            print(f"\nSelecting features for predicting {target}...")
            
            # Prepare the target and feature data
            y = data[target].copy()
            
            # Skip rows with missing target values
            valid_idx = ~y.isna()
            if not valid_idx.all():
                X_valid = X.loc[valid_idx]
                y_valid = y.loc[valid_idx]
            else:
                X_valid = X
                y_valid = y
            
            # Method 1: Mutual Information (works for all kinds of relationships, not just linear)
            try:
                mi_scores = mutual_info_regression(X_valid, y_valid)
                mi_features = pd.Series(mi_scores, index=X_valid.columns)
                mi_features = mi_features.sort_values(ascending=False)
                
                # Keep features with MI score > 0.01 or top 20 features, whichever is smaller
                mi_threshold = max(0.01, mi_features.iloc[min(20, len(mi_features)-1)])
                mi_selected = mi_features[mi_features > mi_threshold].index.tolist()
                
                print(f"  Mutual information selected {len(mi_selected)} features")
                print(f"  Top 5 MI features: {mi_selected[:5]}")
            except Exception as e:
                print(f"  Error in mutual information calculation: {e}")
                mi_selected = []
            
            # Method 2: Feature importance from tree-based model
            try:
                model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
                model.fit(X_valid, y_valid)
                importances = pd.Series(model.feature_importances_, index=X_valid.columns)
                importances = importances.sort_values(ascending=False)
                
                # Keep features with importance > 0.01 or top 20 features
                imp_threshold = max(0.01, importances.iloc[min(20, len(importances)-1)])
                rf_selected = importances[importances > imp_threshold].index.tolist()
                
                print(f"  Random Forest selected {len(rf_selected)} features")
                print(f"  Top 5 RF features: {rf_selected[:5]}")
            except Exception as e:
                print(f"  Error in Random Forest feature selection: {e}")
                rf_selected = []
            
            # Method 3: Recursive Feature Elimination with Cross-Validation
            try:
                if len(X_valid) > 100:  # Only if we have enough data
                    # Start with a simple model for speed
                    base_model = Ridge(alpha=1.0)
                    
                    # Configure RFECV
                    rfecv = RFECV(
                        estimator=base_model,
                        step=1,
                        cv=5,
                        scoring='r2',
                        min_features_to_select=5
                    )
                    
                    # Fit RFECV
                    rfecv.fit(X_valid, y_valid)
                    
                    # Get selected features
                    rfe_selected = X_valid.columns[rfecv.support_].tolist()
                    
                    print(f"  RFE-CV selected {len(rfe_selected)} features")
                    print(f"  Top 5 RFE features: {rfe_selected[:5] if len(rfe_selected) >= 5 else rfe_selected}")
                else:
                    rfe_selected = []
            except Exception as e:
                print(f"  Error in RFECV: {e}")
                rfe_selected = []
            
            # Combine the selected features using a "voting" approach
            feature_votes = {}
            
            for feature in X_valid.columns:
                votes = 0
                if feature in mi_selected:
                    votes += 1
                if feature in rf_selected:
                    votes += 1
                if feature in rfe_selected:
                    votes += 1
                
                feature_votes[feature] = votes
            
            # Select features with at least 2 votes, or the top 15 features if fewer than 10 have 2+ votes
            top_features = [feat for feat, votes in feature_votes.items() if votes >= 2]
            
            if len(top_features) < 10:
                # Sort by votes, then by importance
                sorted_features = sorted(
                    feature_votes.items(),
                    key=lambda x: (x[1], mi_features.get(x[0], 0) if 'mi_features' in locals() else 0),
                    reverse=True
                )
                top_features = [feat for feat, _ in sorted_features[:15]]
            
            print(f"\n  Final selection: {len(top_features)} features for {target}")
            print(f"  Top features: {top_features[:10]}")
            
            # Store the selected features for this target
            selected_features[target] = top_features
    
    # 5.5 Create combined feature set for the final model
    all_selected_features = set()
    for target, features in selected_features.items():
        all_selected_features.update(features)
    
    final_feature_list = list(all_selected_features)
    print(f"\nFinal combined feature set: {len(final_feature_list)} features")
    
    return data, final_feature_list

# 8. Model Evaluation and Visualization
# ===================================

def evaluate_and_visualize_models(models, test_sets, data):
    """
    Perform comprehensive evaluation of all models and create visualizations.
    """
    print("\nPerforming comprehensive model evaluation and visualization...")
    
    # Check if models are available
    if not models:
        print("No models available for evaluation.")
        return None
    
    # Create directory for visualizations
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 8.1 Evaluate each model
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} model...")
        
        # Get corresponding test set
        X_test_key = f"X_test_{model_name}"
        y_test_key = f"y_test_{model_name}"
        
        if X_test_key not in test_sets or y_test_key not in test_sets:
            print(f"Test set not found for {model_name}. Skipping evaluation.")
            continue
        
        X_test = test_sets[X_test_key]
        y_test = test_sets[y_test_key]
        
        # For crop recommendation model (classification)
        if model_name == 'crop_recommendation':
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            print(f"{model_name} Performance Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Confusion Matrix
            plt.figure(figsize=(12, 10))
            classes = np.unique(y_test)
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Normalized Confusion Matrix for Crop Recommendation')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_name}_confusion_matrix.png")
            plt.close()
            
            # Class-wise performance
            class_report = pd.DataFrame(
                precision_recall_fscore_support(y_test, y_pred, labels=classes),
                index=['Precision', 'Recall', 'F1-score', 'Support'],
                columns=classes
            ).T
            
            # Plot class-wise performance
            plt.figure(figsize=(14, len(classes) * 0.4))
            
            # Create horizontal bar chart
            metrics = ['Precision', 'Recall', 'F1-score']
            for i, metric in enumerate(metrics):
                plt.barh(
                    [f"{cls}_{metric}" for cls in classes],
                    class_report[metric].values,
                    height=0.2,
                    label=metric
                )
            
            plt.xlabel('Score')
            plt.title('Class-wise Performance Metrics')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_name}_class_performance.png")
            plt.close()
        
        # For regression models
        else:
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{model_name} Performance Metrics:")
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            
            # Actual vs Predicted Plot
            plt.figure(figsize=(10, 8))
            plt.scatter(y_test, y_pred, alpha=0.5)
            
            # Add diagonal line for perfect predictions
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs. Predicted Values for {model_name}')
            
            # Add metrics annotation
            plt.annotate(
                f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="gray", alpha=0.8)
            )
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_name}_actual_vs_predicted.png")
            plt.close()
            
            # Residual Plot
            residuals = y_test - y_pred
            
            plt.figure(figsize=(10, 8))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot for {model_name}')
            
            # Add RMSE annotation
            plt.annotate(
                f'RMSE = {rmse:.4f}',
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="gray", alpha=0.8)
            )
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_name}_residuals.png")
            plt.close()
            
            # Residual Distribution
            plt.figure(figsize=(10, 8))
            sns.histplot(residuals, kde=True)
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Residuals for {model_name}')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_name}_residual_distribution.png")
            plt.close()
            
            # Error Analysis by Percentile
            if model_name == 'yield_prediction' and 'crop' in data.columns:
                # Get the crop for each test sample
                test_indices = y_test.index
                crops = data.loc[test_indices, 'crop'].values
                
                # Create a dataframe with predictions, errors, and crops
                error_df = pd.DataFrame({
                    'Actual': y_test.values,
                    'Predicted': y_pred,
                    'Error': np.abs(y_test.values - y_pred),
                    'Crop': crops
                })
                
                # Compute average error by crop
                crop_errors = error_df.groupby('Crop')['Error'].mean().sort_values(ascending=False)
                
                # Plot top N crops with highest errors
                plt.figure(figsize=(12, 8))
                crop_errors.head(10).plot(kind='bar')
                plt.ylabel('Mean Absolute Error')
                plt.title(f'Crops with Highest Prediction Errors for {model_name}')
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/{model_name}_crop_errors.png")
                plt.close()
    
    # 8.2 Feature Importance Analysis
    print("\nAnalyzing feature importance across models...")
    
    for model_name, model in models.items():
        # Skip if model doesn't support feature importance
        if not hasattr(model, 'feature_importances_'):
            continue
        
        # Get corresponding test set
        X_test_key = f"X_test_{model_name}"
        
        if X_test_key not in test_sets:
            continue
        
        X_test = test_sets[X_test_key]
        
        # Get feature importance from model
        feature_imp = pd.DataFrame(
            {'Feature': X_test.columns, 'Importance': model.feature_importances_}
        ).sort_values(by='Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_imp.head(20))
        plt.title(f'Top 20 Feature Importance for {model_name} Model')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/{model_name}_feature_importance.png")
        plt.close()
        
        # Calculate permutation importance (more reliable measure)
        # This can be computationally expensive, so we limit to top features
        try:
            # Get test data for this model
            y_test_key = f"y_test_{model_name}"
            if y_test_key not in test_sets:
                continue
                
            y_test = test_sets[y_test_key]
            
            # Calculate permutation importance for top 15 features
            top_features = feature_imp.head(15)['Feature'].tolist()
            X_test_top = X_test[top_features].copy()
            
            # Choose appropriate scoring metric
            if model_name == 'crop_recommendation':
                scoring = 'accuracy'
            else:
                scoring = 'r2'
            
            perm_importance = permutation_importance(
                model, X_test_top, y_test,
                n_repeats=10,
                random_state=RANDOM_SEED,
                scoring=scoring
            )
            
            # Create DataFrame of permutation importance
            perm_imp_df = pd.DataFrame({
                'Feature': top_features,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            }).sort_values(by='Importance', ascending=False)
            
            # Plot permutation importance with error bars
            plt.figure(figsize=(12, 8))
            plt.barh(
                perm_imp_df['Feature'],
                perm_imp_df['Importance'],
                xerr=perm_imp_df['Std'],
                alpha=0.7
            )
            plt.title(f'Permutation Feature Importance for {model_name} Model')
            plt.xlabel('Mean Decrease in Score')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_name}_permutation_importance.png")
            plt.close()
        except Exception as e:
            print(f"Error calculating permutation importance for {model_name}: {e}")
    
    # 8.3 Model Comparison (if we have multiple models)
    if len(models) > 1:
        print("\nComparing models...")
        
        # For regression models, compare R² and RMSE
        regression_models = [name for name in models if name != 'crop_recommendation']
        if len(regression_models) > 1:
            metrics = []
            
            for model_name in regression_models:
                X_test_key = f"X_test_{model_name}"
                y_test_key = f"y_test_{model_name}"
                
                if X_test_key not in test_sets or y_test_key not in test_sets:
                    continue
                
                X_test = test_sets[X_test_key]
                y_test = test_sets[y_test_key]
                
                # Get predictions
                y_pred = models[model_name].predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                metrics.append({
                    'Model': model_name,
                    'R²': r2,
                    'RMSE': rmse,
                    'MAE': mae
                })
            
            # Create comparison dataframe
            if metrics:
                comparison_df = pd.DataFrame(metrics)
                
                # Plot comparison of R²
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x='Model', y='R²', data=comparison_df)
                
                # Add value labels on the bars
                for i, v in enumerate(comparison_df['R²']):
                    ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
                
                plt.title('R² Score Comparison Across Models')
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/model_comparison_r2.png")
                plt.close()
                
                # Plot comparison of RMSE
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x='Model', y='RMSE', data=comparison_df)
                
                # Add value labels on the bars
                for i, v in enumerate(comparison_df['RMSE']):
                    ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
                
                plt.title('RMSE Comparison Across Models')
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/model_comparison_rmse.png")
                plt.close()
    
    # 8.4 Correlation Analysis of Features
    print("\nPerforming correlation analysis of key features...")
    
    # Select relevant columns for correlation analysis
    if 'yield' in data.columns and 'post_harvest_waste_pct' in data.columns:
        # Include target variables and top features for both yield and food loss
        yield_features = []
        waste_features = []
        
        for model_name, model in models.items():
            if not hasattr(model, 'feature_importances_'):
                continue
                
            X_test_key = f"X_test_{model_name}"
            if X_test_key not in test_sets:
                continue
                
            X_test = test_sets[X_test_key]
            
            # Get top features
            features = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            top_features = features.head(10)['Feature'].tolist()
            
            if model_name == 'yield_prediction':
                yield_features = top_features
            elif model_name == 'food_loss':
                waste_features = top_features
        
        # Combine unique features
        corr_features = list(set(yield_features + waste_features))
        
        # Add target variables
        corr_features = ['yield', 'post_harvest_waste_pct'] + corr_features
        
        # Filter to columns that exist in the dataset
        corr_features = [col for col in corr_features if col in data.columns]
        
        # Limit to 20 features for readability
        if len(corr_features) > 20:
            corr_features = corr_features[:20]
        
        # Calculate correlation matrix
        corr_matrix = data[corr_features].corr()
        
        # Plot heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            square=True,
            linewidths=0.5
        )
        plt.title('Correlation Matrix of Key Features')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/correlation_heatmap.png")
        plt.close()
    
    print(f"All visualizations saved to {viz_dir} directory")
    return True


# 9. Perform sensitivity analysis
# ===================================
def perform_sensitivity_analysis(models, test_sets, integrated_data):
    """
    Conduct advanced sensitivity analysis to understand how changes in input parameters
    affect model predictions for yield and food loss.
    """
    print("\nPerforming advanced sensitivity analysis...")
    
    # Check if we have the necessary models
    required_models = ['yield_prediction', 'food_loss']
    
    # Filter to available models
    available_models = [model for model in required_models if model in models]
    
    if not available_models:
        print("No suitable models available for sensitivity analysis.")
        return None
    
    # Create directory for visualizations
    viz_dir = "visualizations/sensitivity"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Parameters to test
    controllable_params = [
        'nitrogen_mean', 'phosphorus_mean', 'potassium_mean',  # Fertilizer
        'ph_mean',  # Soil pH
        'temperature_mean', 'humidity_mean', 'rainfall_mean'  # Environmental
    ]
    
    # Additional engineered features
    if 'npk_balance' in integrated_data.columns:
        controllable_params.append('npk_balance')
    if 'soil_fertility' in integrated_data.columns:
        controllable_params.append('soil_fertility')
    if 'aridity_index' in integrated_data.columns:
        controllable_params.append('aridity_index')
    if 'temp_humidity_index' in integrated_data.columns:
        controllable_params.append('temp_humidity_index')
    
    # Filter to parameters that exist in our data
    available_params = [param for param in controllable_params if param in integrated_data.columns]
    
    if not available_params:
        print("No suitable parameters found for sensitivity analysis.")
        return None
    
    # Define analysis ranges for each parameter
    param_ranges = {}
    for param in available_params:
        # Get parameter min and max values
        min_val = integrated_data[param].min()
        max_val = integrated_data[param].max()
        
        # Set range to cover param distribution
        # (from 10th percentile to 90th percentile to avoid extreme values)
        p10 = integrated_data[param].quantile(0.1)
        p90 = integrated_data[param].quantile(0.9)
        
        if p10 == p90:  # Handle zero variance
            p10 = min_val
            p90 = max_val
        
        param_ranges[param] = np.linspace(p10, p90, 20)
    
    # For each model, analyze sensitivity to each parameter
    sensitivity_results = {}
    
    # Get the base sample for prediction
    for model_name in available_models:
        X_test_key = f"X_test_{model_name}"
        
        if X_test_key not in test_sets:
            print(f"Test set not found for {model_name}. Skipping analysis.")
            continue
        
        X_test = test_sets[X_test_key]
        
        # Use first sample as base (or average of all samples)
        base_sample = X_test.iloc[0].copy()
        
        print(f"\nAnalyzing sensitivity for {model_name} model...")
        sensitivity_results[model_name] = {}
        
        # For each parameter, vary it and check the impact
        for param in available_params:
            if param not in base_sample.index:
                print(f"Parameter {param} not found in test sample. Skipping.")
                continue
                
            print(f"Testing sensitivity to {param}...")
            
            # Get parameter range
            param_values = param_ranges[param]
            predictions = []
            
            # For each value in the range, make a prediction
            for value in param_values:
                # Create a modified sample
                test_sample = base_sample.copy()
                test_sample[param] = value
                
                # Make prediction
                prediction = models[model_name].predict(test_sample.values.reshape(1, -1))[0]
                predictions.append(prediction)
            
            # Store results
            sensitivity_results[model_name][param] = {
                'values': param_values,
                'predictions': predictions,
                # Calculate sensitivity (rate of change)
                'sensitivity': (max(predictions) - min(predictions)) / (param_values[-1] - param_values[0])
            }
            
            # Plot sensitivity curve
            plt.figure(figsize=(10, 6))
            plt.plot(param_values, predictions, marker='o', linestyle='-')
            plt.xlabel(param)
            
            # Add appropriate y-axis label
            if model_name == 'yield_prediction':
                plt.ylabel('Predicted Yield')
                plt.title(f'Sensitivity of Crop Yield to {param}')
            else:  # food_loss
                plt.ylabel('Predicted Food Loss %')
                plt.title(f'Sensitivity of Food Loss to {param}')
            
            # Add trend line
            z = np.polyfit(param_values, predictions, 1)
            p = np.poly1d(z)
            plt.plot(param_values, p(param_values), "r--", alpha=0.8, 
                    label=f"Trend: y = {z[0]:.4f}x + {z[1]:.4f}")
            
            # Calculate and display correlation
            corr = np.corrcoef(param_values, predictions)[0, 1]
            plt.annotate(f"Correlation: {corr:.4f}", xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_name}_{param}_sensitivity.png")
            plt.close()
    
    # Two-way sensitivity analysis for most important parameters
    # Find the most sensitive parameters for each model
    for model_name in sensitivity_results:
        sensitivities = {}
        for param, results in sensitivity_results[model_name].items():
            sensitivities[param] = abs(results['sensitivity'])
        
        # Get the top 3 most sensitive parameters
        top_params = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)[:3]
        top_param_names = [p[0] for p in top_params]
        
        print(f"\nTop sensitive parameters for {model_name}:")
        for param, sensitivity in top_params:
            print(f"  {param}: {sensitivity:.4f}")
        
        # Perform two-way sensitivity analysis for pairs of top parameters
        if len(top_param_names) >= 2:
            for i in range(len(top_param_names) - 1):
                for j in range(i + 1, len(top_param_names)):
                    param1 = top_param_names[i]
                    param2 = top_param_names[j]
                    
                    print(f"Performing two-way sensitivity analysis for {param1} and {param2}...")
                    
                    # Create reduced ranges for performance
                    param1_range = np.linspace(param_ranges[param1][0], param_ranges[param1][-1], 10)
                    param2_range = np.linspace(param_ranges[param2][0], param_ranges[param2][-1], 10)
                    
                    # Initialize grid for predictions
                    predictions_grid = np.zeros((len(param1_range), len(param2_range)))
                    
                    # Make predictions for each combination
                    for i1, val1 in enumerate(param1_range):
                        for i2, val2 in enumerate(param2_range):
                            test_sample = base_sample.copy()
                            test_sample[param1] = val1
                            test_sample[param2] = val2
                            
                            prediction = models[model_name].predict(test_sample.values.reshape(1, -1))[0]
                            predictions_grid[i1, i2] = prediction
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 10))
                    
                    # Plot heatmap
                    im = plt.imshow(predictions_grid, interpolation='bilinear', cmap='viridis', 
                                  aspect='auto', origin='lower')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im)
                    if model_name == 'yield_prediction':
                        cbar.set_label('Predicted Yield')
                    else:
                        cbar.set_label('Predicted Food Loss %')
                    
                    # Set axis labels
                    plt.xlabel(param2)
                    plt.ylabel(param1)
                    
                    # Set tick labels
                    plt.xticks(range(len(param2_range)), 
                              [f"{val:.2f}" for val in param2_range], rotation=45)
                    plt.yticks(range(len(param1_range)), 
                              [f"{val:.2f}" for val in param1_range])
                    
                    plt.title(f'Two-way Sensitivity Analysis: {param1} vs {param2} for {model_name}')
                    plt.tight_layout()
                    plt.savefig(f"{viz_dir}/{model_name}_{param1}_vs_{param2}.png")
                    plt.close()
    
    # Analyze trade-offs between yield and food loss
    if all(model in models for model in ['yield_prediction', 'food_loss']):
        print("\nAnalyzing yield-waste trade-offs...")
        
        # Get test samples
        X_test_yield = test_sets["X_test_yield_prediction"]
        X_test_loss = test_sets["X_test_food_loss"]
        
        # Ensure we have common features
        common_features = [f for f in X_test_yield.columns if f in X_test_loss.columns]
        
        if len(common_features) > 0:
            # Use first sample from each
            base_yield_sample = X_test_yield.iloc[0][common_features].copy()
            base_loss_sample = X_test_loss.iloc[0][common_features].copy()
            
            # For major controllable parameters, analyze trade-offs
            for param in available_params:
                if param not in common_features:
                    continue
                
                print(f"Analyzing yield-waste trade-off for {param}...")
                
                # Get parameter range
                param_values = param_ranges[param]
                yield_predictions = []
                loss_predictions = []
                effective_yield = []  # Yield after accounting for losses
                
                # For each value, predict yield and loss
                for value in param_values:
                    # Create yield sample
                    yield_sample = base_yield_sample.copy()
                    yield_sample[param] = value
                    
                    # Create loss sample
                    loss_sample = base_loss_sample.copy()
                    loss_sample[param] = value
                    
                    # Predict yield and loss
                    try:
                        y_pred = models['yield_prediction'].predict(
                            yield_sample.values.reshape(1, -1))[0]
                        l_pred = models['food_loss'].predict(
                            loss_sample.values.reshape(1, -1))[0]
                        
                        yield_predictions.append(y_pred)
                        loss_predictions.append(l_pred)
                        
                        # Calculate effective yield (accounting for losses)
                        eff_y = y_pred * (1 - l_pred/100)
                        effective_yield.append(eff_y)
                    except Exception as e:
                        print(f"Error predicting for {param}={value}: {e}")
                
                # Plot trade-off
                fig, ax1 = plt.subplots(figsize=(12, 8))
                
                # Plot yield on primary y-axis
                color = 'tab:blue'
                ax1.set_xlabel(param)
                ax1.set_ylabel('Predicted Yield', color=color)
                line1 = ax1.plot(param_values, yield_predictions, color=color, marker='o', 
                                label='Predicted Yield')
                ax1.tick_params(axis='y', labelcolor=color)
                
                # Add effective yield
                line3 = ax1.plot(param_values, effective_yield, color='tab:green', marker='s', 
                                linestyle='--', label='Effective Yield')
                
                # Create secondary y-axis for loss
                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('Predicted Food Loss %', color=color)
                line2 = ax2.plot(param_values, loss_predictions, color=color, marker='x', 
                                label='Predicted Food Loss %')
                ax2.tick_params(axis='y', labelcolor=color)
                
                # Add legend
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper center')
                
                plt.title(f'Yield-Waste Trade-off Analysis for {param}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/trade_off_{param}.png")
                plt.close()
                
                # Calculate optimal value for effective yield
                if effective_yield:
                    opt_idx = np.argmax(effective_yield)
                    opt_value = param_values[opt_idx]
                    opt_yield = yield_predictions[opt_idx]
                    opt_loss = loss_predictions[opt_idx]
                    opt_effective = effective_yield[opt_idx]
                    
                    print(f"  Optimal {param} value: {opt_value:.4f}")
                    print(f"  Expected yield: {opt_yield:.2f}")
                    print(f"  Expected loss: {opt_loss:.2f}%")
                    print(f"  Effective yield: {opt_effective:.2f}")
    
    return sensitivity_results


# 10. Create prediction system
# ===================================
def create_prediction_system(models, data, feature_list):
    """
    Create a prediction system for practical use of the models.
    
    Args:
        models: Dictionary of trained models
        data: Integrated dataset
        feature_list: List of features used in the models
    
    Returns:
        A prediction function that can be used to make predictions
    """
    print("\nCreating prediction system for practical use...")
    
    # Save all models and metadata
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save each model
    for model_name, model in models.items():
        model_path = f"{model_dir}/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved {model_name} model to {model_path}")
    
    # Save feature list
    feature_list_path = f"{model_dir}/feature_list.pkl"
    joblib.dump(feature_list, feature_list_path)
    
    # Create system metadata
    metadata = {
        'models': list(models.keys()),
        'feature_list': feature_list,
        'data_columns': list(data.columns),
        'categorical_features': list(data.select_dtypes(include=['object', 'category']).columns),
        'numeric_features': list(data.select_dtypes(include=[np.number]).columns),
        'feature_ranges': {
            col: (data[col].min(), data[col].max()) 
            for col in data.select_dtypes(include=[np.number]).columns
        }
    }
    
    # Save metadata
    metadata_path = f"{model_dir}/metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"Saved system metadata to {metadata_path}")
    
    # Create prediction function
    def prediction_system(input_params, target_crop=None):
        """
        Make predictions using the trained models.
        
        Args:
            input_params: Dictionary of input parameters
                (geographical, environmental, temporal)
            target_crop: Optional; if provided, will optimize for this crop,
                otherwise will recommend the best crop
                
        Returns:
            Dictionary of predictions and recommendations
        """
        # Load models and metadata
        loaded_models = {}
        for model_name in metadata['models']:
            model_path = f"{model_dir}/{model_name}.pkl"
            loaded_models[model_name] = joblib.load(model_path)
        
        # Preprocess input parameters
        processed_input = {}
        
        # Fill missing parameters with defaults from data
        for feature in feature_list:
            if feature in input_params:
                processed_input[feature] = input_params[feature]
            elif feature in metadata['feature_ranges']:
                # Use median value as default
                min_val, max_val = metadata['feature_ranges'][feature]
                processed_input[feature] = (min_val + max_val) / 2
        
        # Handle crop recommendation or optimization
        if 'crop_recommendation' in loaded_models and target_crop is None:
            # Create feature vector for crop recommendation
            crop_features = [f for f in feature_list if f in processed_input]
            X_crop = pd.DataFrame([{f: processed_input[f] for f in crop_features}])
            
            # Recommend crop
            recommended_crop = loaded_models['crop_recommendation'].predict(X_crop)[0]
            processed_input['crop'] = recommended_crop
        elif target_crop is not None:
            processed_input['crop'] = target_crop
        
        # One-hot encode crop if needed
        crop_cols = [col for col in feature_list if col.startswith('crop_')]
        if crop_cols and 'crop' in processed_input:
            for col in crop_cols:
                processed_input[col] = 0
            
            target_col = f"crop_{processed_input['crop']}"
            if target_col in feature_list:
                processed_input[target_col] = 1
        
        # Make predictions
        predictions = {}
        
        # Create feature vector for prediction models
        X_pred = pd.DataFrame([{f: processed_input.get(f, 0) for f in feature_list}])
        
        # Predict yield
        if 'yield_prediction' in loaded_models:
            try:
                yield_pred = loaded_models['yield_prediction'].predict(X_pred)[0]
                predictions['yield'] = yield_pred
            except Exception as e:
                print(f"Error predicting yield: {e}")
        
        # Predict food loss
        if 'food_loss' in loaded_models:
            try:
                loss_pred = loaded_models['food_loss'].predict(X_pred)[0]
                predictions['food_loss_pct'] = loss_pred
            except Exception as e:
                print(f"Error predicting food loss: {e}")
        
        # Calculate effective yield
        if 'yield' in predictions and 'food_loss_pct' in predictions:
            predictions['effective_yield'] = predictions['yield'] * (1 - predictions['food_loss_pct']/100)
        
        # Add recommendations for controllable parameters
        if 'yield_prediction' in loaded_models and 'food_loss' in loaded_models:
            try:
                # Define controllable parameters
                controllable_params = [
                    'nitrogen_mean', 'phosphorus_mean', 'potassium_mean',
                    'ph_mean', 'rainfall_mean'
                ]
                
                # Filter to available parameters
                available_params = [p for p in controllable_params if p in feature_list]
                
                # Simple optimization for each parameter
                optimizations = {}
                
                for param in available_params:
                    # Get parameter range
                    if param in metadata['feature_ranges']:
                        min_val, max_val = metadata['feature_ranges'][param]
                        param_range = np.linspace(min_val, max_val, 20)
                        
                        best_value = None
                        best_score = float('-inf')
                        
                        for value in param_range:
                            # Create test sample
                            test_input = X_pred.copy()
                            test_input.loc[0, param] = value
                            
                            # Predict yield and loss
                            y_pred = loaded_models['yield_prediction'].predict(test_input)[0]
                            l_pred = loaded_models['food_loss'].predict(test_input)[0]
                            
                            # Calculate effective yield
                            eff_yield = y_pred * (1 - l_pred/100)
                            
                            # Update best value
                            if eff_yield > best_score:
                                best_score = eff_yield
                                best_value = value
                        
                        if best_value is not None:
                            optimizations[param] = {
                                'recommended_value': best_value,
                                'expected_impact': best_score - predictions.get('effective_yield', 0)
                            }
                
                # Add to predictions
                predictions['parameter_recommendations'] = optimizations
            except Exception as e:
                print(f"Error generating parameter recommendations: {e}")
        
        # Complete response
        response = {
            'input_parameters': input_params,
            'recommended_crop': processed_input.get('crop') if target_crop is None else target_crop,
            'predictions': predictions,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return response
    
    # Test the prediction system
    print("\nTesting prediction system with sample input...")
    
    # Create sample input
    sample_input = {}
    
    # Add environmental parameters
    env_params = ['temperature_mean', 'humidity_mean', 'rainfall_mean', 'ph_mean']
    for param in env_params:
        if param in data.columns:
            sample_input[param] = data[param].median()
    
    # Add nutrient parameters
    nutrient_params = ['nitrogen_mean', 'phosphorus_mean', 'potassium_mean']
    for param in nutrient_params:
        if param in data.columns:
            sample_input[param] = data[param].median()
    
    # Test prediction
    try:
        test_prediction = prediction_system(sample_input)
        print("\nSample prediction result:")
        for key, value in test_prediction.items():
            if key == 'predictions':
                print(f"{key}:")
                for pred_key, pred_value in value.items():
                    if pred_key != 'parameter_recommendations':
                        print(f"  {pred_key}: {pred_value:.4f}")
            elif key != 'parameter_recommendations':
                print(f"{key}: {value}")
        
        # If recommendations available, show a sample
        if 'parameter_recommendations' in test_prediction['predictions']:
            print("\nSample parameter recommendations:")
            for param, details in list(test_prediction['predictions']['parameter_recommendations'].items())[:3]:
                print(f"  {param}: {details['recommended_value']:.4f} (impact: {details['expected_impact']:.4f})")
    except Exception as e:
        print(f"Error testing prediction system: {e}")
    
    print("\nPrediction system created successfully.")
    return prediction_system


# 6. Model Development
# ==================

def develop_models(data, feature_list, n_trials=50):
    """
    Develop advanced ML models for crop recommendation, yield prediction, 
    and food loss estimation with extensive hyperparameter optimization.
    
    Args:
        data: The integrated and feature-engineered dataset
        feature_list: List of selected features to use
        n_trials: Number of hyperparameter optimization trials
        
    Returns:
        Dictionary of trained models and evaluation metrics
    """
    print("\nDeveloping advanced machine learning models...")
    
    if data is None or feature_list is None or len(feature_list) == 0:
        print("Insufficient data or features for model development.")
        return None, None, None
    
    # Initialize containers for results
    models = {}
    results = {}
    test_sets = {}
    
    # Check if we have the necessary target columns
    target_columns = {
        'crop_recommendation': 'crop',
        'yield_prediction': 'yield',
        'food_loss': 'post_harvest_waste_pct'
    }
    
    available_targets = {k: v for k, v in target_columns.items() if v in data.columns}
    
    if not available_targets:
        print("No valid target columns found in the dataset.")
        return None, None, None
    
    # Use only available features from feature_list
    available_features = [feat for feat in feature_list if feat in data.columns]
    
    if len(available_features) < 2:
        print("Not enough features available for modeling.")
        return None, None, None
    
    print(f"Using {len(available_features)} features for modeling:")
    print(f"Sample features: {available_features[:5]}...")
    
    # 6.1 Model Development for each target
    for model_name, target_col in available_targets.items():
        print(f"\nDeveloping {model_name} model...")
        
        # Extract the target
        y = data[target_col].copy()
        
        # Check if enough data is available
        if y.isna().sum() > 0.5 * len(y):
            print(f"Too many missing values in target '{target_col}'. Skipping this model.")
            continue
        
        # Remove rows with missing target values
        valid_idx = ~y.isna()
        X_filtered = data.loc[valid_idx, available_features].copy()
        y_filtered = y.loc[valid_idx].copy()
        
        # Check if we still have enough data
        if len(X_filtered) < 100:
            print(f"Not enough data for {model_name} model after filtering. Need at least 100 rows.")
            continue
        
        print(f"Proceeding with {len(X_filtered)} samples for {model_name} model.")
        
        # For crop recommendation model (classification)
        if model_name == 'crop_recommendation':
            # Filter classes that have enough samples
            class_counts = y_filtered.value_counts()
            valid_classes = class_counts[class_counts >= 5].index  # Keep only classes with >=5 samples
            X_filtered = X_filtered[y_filtered.isin(valid_classes)]
            y_filtered = y_filtered[y_filtered.isin(valid_classes)]

            label_encoder = LabelEncoder()
            y_filtered = pd.Series(label_encoder.fit_transform(y_filtered))
            
            os.makedirs("models", exist_ok=True)
            joblib.dump(label_encoder, "models/crop_label_encoder.pkl")
            
            # Check if all classes have at least 2 samples for stratify
            if y_filtered.value_counts().min() >= 2:
                stratify_option = y_filtered
            else:
                print(f"Warning: Some classes have less than 2 samples. Disabling stratify for {model_name}.")
                stratify_option = None

            print(f"Keeping {len(valid_classes)} crops out of {len(class_counts)} after filtering rare crops.")


            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.25, random_state=RANDOM_SEED, stratify=stratify_option
            )

                    
            # Save test set for later evaluation
            test_sets[f"X_test_{model_name}"] = X_test
            test_sets[f"y_test_{model_name}"] = y_test
            
            # Hyperparameter optimization with Optuna
            print(f"Optimizing hyperparameters for {model_name} model with {n_trials} trials...")
            
            def objective(trial):
                # Try different classifier algorithms
                classifier_name = trial.suggest_categorical('classifier', [
                    'random_forest', 'gradient_boosting', 'xgboost'
                ])
                
                if classifier_name == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'random_state': RANDOM_SEED
                    }
                    model = RandomForestClassifier(**params)
                
                elif classifier_name == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'random_state': RANDOM_SEED
                    }
                    model = xgb.XGBClassifier(**params)
                
                else:  # gradient_boosting
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'random_state': RANDOM_SEED
                    }
                    model = GradientBoostingClassifier(**params)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=5, scoring='f1_weighted'
                )
                
                # Return the mean F1 score
                return cv_scores.mean()
            
            # Create and run the study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Get the best parameters
            best_params = study.best_params
            best_classifier = best_params.pop('classifier')
            
            print(f"Best algorithm: {best_classifier}")
            print(f"Best parameters: {best_params}")
            
            # Train the final model with the best parameters
            if best_classifier == 'random_forest':
                final_model = RandomForestClassifier(**best_params)
            elif best_classifier == 'xgboost':
                final_model = xgb.XGBClassifier(**best_params)
            else:  # gradient_boosting
                final_model = GradientBoostingClassifier(**best_params)
            
            # Fit on the full training set
            final_model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = final_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            print("\nCrop Recommendation Model Performance:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Store results
            models[model_name] = final_model
            results[model_name] = {
                'algorithm': best_classifier,
                'parameters': best_params,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            }
        
        # For regression models (yield prediction and food loss)
        else:
            # Split data for regression
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.25, random_state=RANDOM_SEED
            )
            
            # Save test set for later evaluation
            test_sets[f"X_test_{model_name}"] = X_test
            test_sets[f"y_test_{model_name}"] = y_test
            
            # Hyperparameter optimization with Optuna
            print(f"Optimizing hyperparameters for {model_name} model with {n_trials} trials...")
            
            def objective(trial):
                # Try different regression algorithms
                regressor_name = trial.suggest_categorical('regressor', [
                    'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'
                ])
                
                if regressor_name == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'random_state': RANDOM_SEED
                    }
                    model = RandomForestRegressor(**params)
                
                elif regressor_name == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'random_state': RANDOM_SEED
                    }
                    model = xgb.XGBRegressor(**params)
                
                elif regressor_name == 'lightgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                        'random_state': RANDOM_SEED
                    }
                    model = lgb.LGBMRegressor(**params)
                
                else:  # gradient_boosting
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'random_state': RANDOM_SEED
                    }
                    model = GradientBoostingRegressor(**params)
                
                # Cross-validation with appropriate scoring
                if model_name == 'yield_prediction':
                    scoring = 'neg_root_mean_squared_error'
                else:  # food_loss
                    scoring = 'r2'
                
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=5, scoring=scoring
                )
                
                # Return the mean score
                return cv_scores.mean()
            
            # Create and run the study
            direction = 'maximize' if model_name != 'yield_prediction' else 'minimize'
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=n_trials)
            
            # Get the best parameters
            best_params = study.best_params
            best_regressor = best_params.pop('regressor')
            
            print(f"Best algorithm: {best_regressor}")
            print(f"Best parameters: {best_params}")
            
            # Train the final model with the best parameters
            if best_regressor == 'random_forest':
                final_model = RandomForestRegressor(**best_params)
            elif best_regressor == 'xgboost':
                final_model = xgb.XGBRegressor(**best_params)
            elif best_regressor == 'lightgbm':
                final_model = lgb.LGBMRegressor(**best_params)
            else:  # gradient_boosting
                final_model = GradientBoostingRegressor(**best_params)
            
            # Fit on the full training set
            final_model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = final_model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"\n{model_name} Model Performance:")
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            
            # Store results
            models[model_name] = final_model
            results[model_name] = {
                'algorithm': best_regressor,
                'parameters': best_params,
                'metrics': {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae
                }
            }
    
    # 6.2 Create visualization of model performance
    print("\nGenerating model performance visualizations...")
    
    # Create directory for visualizations
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize model performance metrics
    model_metrics = {}
    for model_name, result in results.items():
        metrics = result['metrics']
        model_metrics[model_name] = metrics
    
    # Visualize metrics for each model type
    for model_type in ['crop_recommendation', 'yield_prediction', 'food_loss']:
        if model_type in model_metrics:
            metrics = model_metrics[model_type]
            
            plt.figure(figsize=(10, 6))
            plt.bar(metrics.keys(), metrics.values())
            plt.title(f'{model_type} Model Performance Metrics')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_type}_metrics.png")
            plt.close()
    
    # Cross-validation performance tracking if available
    for model_name, model in models.items():
        if model_name in available_targets:
            target = available_targets[model_name]
            X = data.loc[~data[target].isna(), available_features].copy()
            y = data.loc[~data[target].isna(), target].copy()
            
            # Perform 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            fold_scores = []
            
            for i, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train a model on this fold
                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Evaluate
                if model_name == 'crop_recommendation':
                    fold_score = accuracy_score(y_fold_val, fold_model.predict(X_fold_val))
                    metric_name = 'Accuracy'
                else:
                    fold_score = r2_score(y_fold_val, fold_model.predict(X_fold_val))
                    metric_name = 'R² Score'
                
                fold_scores.append(fold_score)
            
            # Plot cross-validation results
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, 6), fold_scores, marker='o')
            plt.axhline(y=np.mean(fold_scores), color='r', linestyle='--', label=f'Mean: {np.mean(fold_scores):.4f}')
            plt.title(f'{model_name} 5-Fold Cross-Validation Performance')
            plt.xlabel('Fold')
            plt.ylabel(metric_name)
            plt.xticks(range(1, 6))
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_name}_cv_performance.png")
            plt.close()
    
    # Feature importance visualization for each model
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features = [available_features[i] for i in indices]
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance for {model_name} Model')
            plt.barh(range(min(20, len(features))), importances[indices][:20], align='center')
            plt.yticks(range(min(20, len(features))), [features[i] for i in range(min(20, len(features)))])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{model_name}_feature_importance.png")
            plt.close()
    
    print(f"Model visualizations saved to {viz_dir} directory")
    
    return models, results, test_sets

# 7. Agricultural Parameter Optimization
# ====================================

def optimize_agricultural_parameters(models, test_sets, data, available_features, n_iterations=100):
    """
    Use advanced optimization techniques to identify optimal agricultural parameters.
    
    Args:
        models: Dictionary of trained models
        test_sets: Dictionary of test datasets
        data: Integrated dataset
        available_features: List of available features
        n_iterations: Number of optimization iterations
        
    Returns:
        Dictionary of optimized parameters for different scenarios
    """
    print("\nPerforming advanced agricultural parameter optimization...")
    
    # Check if we have the necessary models
    required_models = ['yield_prediction', 'food_loss']
    if not all(model in models for model in required_models):
        print("Missing required models for optimization. Skipping this step.")
        return None
    
    # Identify controllable agricultural parameters
    # These are parameters that farmers could realistically adjust
    controllable_params = [
        'nitrogen_mean', 'phosphorus_mean', 'potassium_mean',  # Fertilizer
        'ph_mean',  # Soil amendments
        'rainfall_mean',  # Irrigation (as a proxy)
        'crop'  # Crop selection (if not specified)
    ]
    
    # Also include any engineered features related to these
    for feature in available_features:
        if any(param in feature for param in ['nitrogen', 'phosphorus', 'potassium', 'ph', 'rainfall']):
            controllable_params.append(feature)
    
    # Filter to parameters that exist in our dataset
    controllable_params = [param for param in controllable_params if param in data.columns]
    
    # Determine parameter bounds for optimization
    param_bounds = {}
    for param in controllable_params:
        if param == 'crop':
            # For crop, we'll handle separately
            continue
        elif 'ratio' in param:
            # For ratio features, bound between 0 and 1
            param_bounds[param] = (0, 1)
        else:
            # For numerical parameters, use data range with slight expansion
            min_val = data[param].min()
            max_val = data[param].max()
            range_val = max_val - min_val
            
            # Expand bounds slightly to allow exploration
            param_bounds[param] = (max(0, min_val - 0.1 * range_val), max_val + 0.1 * range_val)
    
    print(f"Optimizing {len(controllable_params)} agricultural parameters:")
    for param, bounds in param_bounds.items():
        print(f"  {param}: {bounds}")
    
    # Create optimization scenarios
    scenarios = []
    
    # Scenario 1: Maximize yield and minimize waste with fixed crop
    # We'll do this for each major crop in the dataset
    if 'crop' in data.columns:
        top_crops = data['crop'].value_counts().head(5).index.tolist()
        
        for crop in top_crops:
            scenarios.append({
                'name': f"optimize_for_{crop}",
                'description': f"Maximize yield and minimize waste for {crop}",
                'fixed_params': {'crop': crop},
                'objective': 'yield_minus_waste'
            })
    
    # Scenario 2: Maximize yield with no crop constraint
    scenarios.append({
        'name': "maximize_yield",
        'description': "Maximize yield without crop constraint",
        'fixed_params': {},
        'objective': 'yield'
    })
    
    # Scenario 3: Minimize waste with no crop constraint
    scenarios.append({
        'name': "minimize_waste",
        'description': "Minimize post-harvest waste without crop constraint",
        'fixed_params': {},
        'objective': 'waste'
    })
    
    # Scenario 4: Balanced optimization (yield vs. waste)
    scenarios.append({
        'name': "balanced_optimization",
        'description': "Balance yield maximization and waste minimization",
        'fixed_params': {},
        'objective': 'balanced'
    })
    
    # Run optimization for each scenario
    optimization_results = {}
    
    for scenario in scenarios:
        print(f"\nRunning optimization for scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        # Prepare the optimization space
        opt_params = {p: param_bounds[p] for p in param_bounds if p not in scenario['fixed_params']}
        
        # Define the objective function for this scenario
        def objective_function(**params):
            # Create a sample input for prediction
            if any(f"X_test_yield_prediction" in key for key in test_sets.keys()):
                sample = test_sets["X_test_yield_prediction"].iloc[0:1].copy()
            else:
                # If no test set available, use a random row from the dataset
                sample_features = [col for col in available_features if col in data.columns]
                sample = data.sample(1)[sample_features].copy()
            
            # Update with fixed parameters for this scenario
            for param, value in scenario['fixed_params'].items():
                if param in sample.columns:
                    sample[param] = value
                elif param == 'crop':
                    # Handle one-hot encoded crop columns if they exist
                    crop_cols = [col for col in sample.columns if col.startswith('crop_')]
                    if crop_cols:
                        for col in crop_cols:
                            sample[col] = 0
                        # Try to set the correct crop column to 1
                        crop_col = f"crop_{value}"
                        if crop_col in sample.columns:
                            sample[crop_col] = 1
            
            # Update with the parameters we're optimizing
            for param, value in params.items():
                if param in sample.columns:
                    sample[param] = value
            
            # Predict yield and food loss
            try:
                predicted_yield = models['yield_prediction'].predict(sample)[0]
                predicted_loss = models['food_loss'].predict(sample)[0]
                
                # Calculate objective based on scenario
                if scenario['objective'] == 'yield':
                    # Maximize yield
                    objective_score = predicted_yield
                elif scenario['objective'] == 'waste':
                    # Minimize waste (negative for maximization problem)
                    objective_score = -predicted_loss
                elif scenario['objective'] == 'yield_minus_waste':
                    # Maximize difference
                    # Normalize to make comparable
                    max_yield = data['yield'].max()
                    max_loss = data['post_harvest_waste_pct'].max() if 'post_harvest_waste_pct' in data.columns else 100
                    
                    norm_yield = predicted_yield / max_yield
                    norm_loss = predicted_loss / max_loss
                    
                    objective_score = norm_yield - norm_loss
                else:  # balanced
                    # Weighted combination
                    max_yield = data['yield'].max()
                    max_loss = data['post_harvest_waste_pct'].max() if 'post_harvest_waste_pct' in data.columns else 100
                    
                    norm_yield = predicted_yield / max_yield
                    norm_loss = 1 - (predicted_loss / max_loss)  # Invert so higher is better
                    
                    objective_score = 0.7 * norm_yield + 0.3 * norm_loss
                
                return objective_score
            except Exception as e:
                print(f"Error in objective function: {e}")
                return -999  # Penalty for errors
        
        # Run Bayesian Optimization
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=opt_params,
            random_state=RANDOM_SEED,
            verbose=2
        )
        
        # Perform optimization
        optimizer.maximize(
            init_points=5,
            n_iter=n_iterations
        )
        
        # Get the best parameters
        best_params = optimizer.max['params']
        
        # Add fixed parameters
        for param, value in scenario['fixed_params'].items():
            best_params[param] = value
        
        # Calculate expected yield and food loss with optimal parameters
        # Create prediction sample
        if any(f"X_test_yield_prediction" in key for key in test_sets.keys()):
            sample = test_sets["X_test_yield_prediction"].iloc[0:1].copy()
        else:
            sample_features = [col for col in available_features if col in data.columns]
            sample = data.sample(1)[sample_features].copy()
        
        # Update with optimal parameters
        for param, value in best_params.items():
            if param in sample.columns:
                sample[param] = value
            elif param == 'crop':
                # Handle one-hot encoded crop columns
                crop_cols = [col for col in sample.columns if col.startswith('crop_')]
                if crop_cols:
                    for col in crop_cols:
                        sample[col] = 0
                    crop_col = f"crop_{value}"
                    if crop_col in sample.columns:
                        sample[crop_col] = 1
        
        # Make predictions
        try:
            expected_yield = models['yield_prediction'].predict(sample)[0]
            expected_loss = models['food_loss'].predict(sample)[0]
            
            print(f"Expected Yield with Optimal Parameters: {expected_yield:.2f}")
            print(f"Expected Post-Harvest Loss with Optimal Parameters: {expected_loss:.2f}%")
            print(f"Effective Yield (after losses): {expected_yield * (1 - expected_loss/100):.2f}")
            
            # Store results
            optimization_results[scenario['name']] = {
                'parameters': best_params,
                'expected_yield': expected_yield,
                'expected_loss': expected_loss,
                'effective_yield': expected_yield * (1 - expected_loss/100)
            }
        except Exception as e:
            print(f"Error making predictions with optimal parameters: {e}")
    
    # Create visualization of optimization results
    print("\nCreating visualization of optimization results...")
    
    # Compare expected yield and loss across scenarios
    if optimization_results:
        scenarios = list(optimization_results.keys())
        yields = [optimization_results[s]['expected_yield'] for s in scenarios]
        losses = [optimization_results[s]['expected_loss'] for s in scenarios]
        effective_yields = [optimization_results[s]['effective_yield'] for s in scenarios]
        
        # Create a comparison plot
        plt.figure(figsize=(14, 8))
        
        # Set width of bars
        barWidth = 0.25
        
        # Set positions of the bars on X axis
        r1 = np.arange(len(scenarios))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        
        # Create bars
        plt.bar(r1, yields, width=barWidth, label='Expected Yield')
        plt.bar(r2, effective_yields, width=barWidth, label='Effective Yield (after losses)')
        plt.bar(r3, losses, width=barWidth, label='Expected Loss %')
        
        # Add labels and legend
        plt.xlabel('Optimization Scenario')
        plt.ylabel('Value')
        plt.title('Comparison of Optimization Results Across Scenarios')
        plt.xticks([r + barWidth for r in range(len(scenarios))], [s.replace('_', ' ').title() for s in scenarios], rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("visualizations/optimization_comparison.png")
        plt.close()
        
        # Create detailed parameter plots for each scenario
        for scenario_name, result in optimization_results.items():
            params = result['parameters']
            
            # Skip if not enough parameters
            if len(params) < 2:
                continue
            
            # Create a radar chart for parameter values
            param_names = list(params.keys())
            param_values = list(params.values())
            
            # Normalize parameter values to 0-1 scale for radar chart
            normalized_values = []
            for i, param in enumerate(param_names):
                if param in param_bounds:
                    min_val, max_val = param_bounds[param]
                    range_val = max_val - min_val
                    if range_val > 0:
                        normalized_values.append((param_values[i] - min_val) / range_val)
                    else:
                        normalized_values.append(0.5)  # Default for zero range
                else:
                    normalized_values.append(0.5)  # Default for parameters without bounds
            
            # Create radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of variables
            N = len(param_names)
            
            # Angle of each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add parameter values
            normalized_values += normalized_values[:1]  # Close the loop
            
            # Draw the plot
            ax.plot(angles, normalized_values, linewidth=2, linestyle='solid')
            ax.fill(angles, normalized_values, alpha=0.25)
            
            # Add labels
            plt.xticks(angles[:-1], param_names)
            
            # Add title
            plt.title(f'Optimal Parameters for {scenario_name.replace("_", " ").title()}')
            
            plt.tight_layout()
            plt.savefig(f"visualizations/{scenario_name}_parameters.png")
            plt.close()
    
    return optimization_results
   


# 11. Final Pipeline and Main Function
# ==================================

def main(n_trials=50, n_iterations=100, fast_mode=False):
    """
    Main function to run the entire analysis pipeline.
    
    Args:
        n_trials: Number of hyperparameter optimization trials
        n_iterations: Number of agricultural parameter optimization iterations
        
    Returns:
        Dictionary of results including models, optimized parameters, and prediction system
    """
    print("\n" + "=" * 80)
    print("STARTING AGRICULTURAL FOOD WASTE ANALYSIS PIPELINE")
    print("=" * 80)
    
    start_time = time.time()

    if fast_mode:
        print("FAST MODE ENABLED: Reducing trials and iterations.")
        n_trials = 5
        n_iterations = 10
    
    # Step 1: Download and prepare data
    print("\n" + "=" * 50)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 50)
    fao_data, crop_rec_data, crop_yield_data = download_and_prepare_data()
    
    # Step 2: Clean data
    print("\n" + "=" * 50)
    print("STEP 2: DATA CLEANING AND VALIDATION")
    print("=" * 50)
    fao_clean, crop_rec_clean, crop_yield_clean = clean_data(
        fao_data, crop_rec_data, crop_yield_data
    )
    
    # Step 3: Transform and normalize data
    print("\n" + "=" * 50)
    print("STEP 3: DATA TRANSFORMATION AND NORMALIZATION")
    print("=" * 50)
    fao_transformed, crop_rec_transformed, crop_yield_transformed = transform_and_normalize_data(
        fao_clean, crop_rec_clean, crop_yield_clean
    )
    
    # Step 4: Integrate datasets
    print("\n" + "=" * 50)
    print("STEP 4: DATASET INTEGRATION")
    print("=" * 50)
    integrated_data = integrate_datasets(
        fao_transformed, crop_rec_transformed, crop_yield_transformed
    )
    
    if integrated_data is None:
        print("Dataset integration failed. Exiting pipeline.")
        return None
    
    if fast_mode and integrated_data is not None and len(integrated_data) > 500:
        integrated_data = integrated_data.sample(500, random_state=RANDOM_SEED)
    
    # Step 5: Feature engineering and selection
    print("\n" + "=" * 50)
    print("STEP 5: FEATURE ENGINEERING AND SELECTION")
    print("=" * 50)
    engineered_data, selected_features = engineer_and_select_features(integrated_data)
    
    # Step 6: Develop models
    print("\n" + "=" * 50)
    print("STEP 6: MODEL DEVELOPMENT WITH HYPERPARAMETER OPTIMIZATION")
    print("=" * 50)
    models, results, test_sets = develop_models(engineered_data, selected_features, n_trials=n_trials)
    
    # Step 7: Optimize agricultural parameters
    print("\n" + "=" * 50)
    print("STEP 7: AGRICULTURAL PARAMETER OPTIMIZATION")
    print("=" * 50)
    optimal_params = optimize_agricultural_parameters(
        models, test_sets, engineered_data, selected_features, n_iterations=n_iterations
    )

    # Step 7.5: Compare optimized vs unoptimized effective yields (Dataset approach)
    print("\n" + "=" * 50)
    print("STEP 7.5: MODEL EFFECTIVENESS CALCULATION")
    print("=" * 50)
    
    # NEW: Survival comparison
    survival_dir = "summary/survival"
    os.makedirs(survival_dir, exist_ok=True)
    all_survival_data = []

    if optimal_params and "X_test_yield_prediction" in test_sets:
        X_test_yield = test_sets["X_test_yield_prediction"].copy()
        yield_model = models.get('yield_prediction')
        loss_model = models.get('food_loss')

        for scenario_name, result in optimal_params.items():
            survival_data = []
            optimized_effective_yield = result.get('effective_yield', None)

            for idx in range(len(X_test_yield)):
                unoptimized_input = X_test_yield.iloc[idx:idx+1]

                try:
                    unoptimized_yield = yield_model.predict(unoptimized_input)[0]
                    unoptimized_loss = loss_model.predict(unoptimized_input)[0]
                    unoptimized_effective_yield = unoptimized_yield * (1 - unoptimized_loss / 100)

                    if optimized_effective_yield and unoptimized_effective_yield > 0:
                        survival_percentage = (optimized_effective_yield / unoptimized_effective_yield) * 100

                        survival_data.append({
                            'test_index': idx,
                            'unoptimized_effective_yield': unoptimized_effective_yield,
                            'optimized_effective_yield': optimized_effective_yield,
                            'survival_percentage': survival_percentage
                        })
                except:
                    continue

            if survival_data:
                scenario_df = pd.DataFrame(survival_data)
                scenario_filename = f"{survival_dir}/{scenario_name}_survival.csv"
                scenario_df.to_csv(scenario_filename, index=False)
                print(f"Saved survival analysis for scenario '{scenario_name}'")
                scenario_df['scenario'] = scenario_name
                all_survival_data.append(scenario_df)

    # Merge and summarize
    if all_survival_data:
        combined_survival_df = pd.concat(all_survival_data, ignore_index=True)
        effectiveness_summary = combined_survival_df.groupby('scenario')['survival_percentage'].mean().sort_values(ascending=False)

        print("\nModel Effectiveness Summary:")
        print(effectiveness_summary)

        model_effectiveness = effectiveness_summary.mean()
        print(f"\nOverall Model Effectiveness: {model_effectiveness:.2f}%")
    else:
        model_effectiveness = None

    
    # Step 8: Evaluate and visualize models
    print("\n" + "=" * 50)
    print("STEP 8: MODEL EVALUATION AND VISUALIZATION")
    print("=" * 50)
    evaluate_and_visualize_models(models, test_sets, engineered_data)
    
    # Step 9: Perform sensitivity analysis
    print("\n" + "=" * 50)
    print("STEP 9: SENSITIVITY ANALYSIS")
    print("=" * 50)
    sensitivity_results = perform_sensitivity_analysis(models, test_sets, engineered_data)
    
    # Step 10: Create prediction system
    print("\n" + "=" * 50)
    print("STEP 10: MODEL DEPLOYMENT AND PREDICTION SYSTEM")
    print("=" * 50)
    prediction_system = create_prediction_system(models, engineered_data, selected_features)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_time:.2f} SECONDS")
    print("=" * 80)
    
    # Generate summary report
    print("\nGenerating summary report...")
    
    # Create summary directory
    summary_dir = "summary"
    os.makedirs(summary_dir, exist_ok=True)
    
    # Model performance summary
    model_summary = {}
    for model_name, result in results.items():
        model_summary[model_name] = {
            'algorithm': result.get('algorithm', 'Unknown'),
            'metrics': result.get('metrics', {})
        }
    
    # Save model summary
    with open(f"{summary_dir}/model_summary.txt", "w") as f:
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("=========================\n\n")
        
        for model_name, summary in model_summary.items():
            f.write(f"{model_name.upper()}\n")
            f.write(f"Algorithm: {summary['algorithm']}\n")
            f.write("Metrics:\n")
            
            for metric, value in summary['metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\n")
    
    # Optimization summary
    if optimal_params:
        with open(f"{summary_dir}/optimization_summary.txt", "w") as f:
            f.write("AGRICULTURAL PARAMETER OPTIMIZATION SUMMARY\n")
            f.write("==========================================\n\n")
            
            for scenario_name, result in optimal_params.items():
                f.write(f"Scenario: {scenario_name}\n")
                f.write(f"Expected Yield: {result.get('expected_yield', 'N/A'):.4f}\n")
                f.write(f"Expected Loss: {result.get('expected_loss', 'N/A'):.4f}%\n")
                f.write(f"Effective Yield: {result.get('effective_yield', 'N/A'):.4f}\n")
                
                f.write("Optimal Parameters:\n")
                for param, value in result.get('parameters', {}).items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {param}: {value:.4f}\n")
                    else:
                        f.write(f"  {param}: {value}\n")
                
                f.write("\n")
    
    # Sensitivity summary
    if sensitivity_results:
        with open(f"{summary_dir}/sensitivity_summary.txt", "w") as f:
            f.write("SENSITIVITY ANALYSIS SUMMARY\n")
            f.write("===========================\n\n")
            
            for model_name, params in sensitivity_results.items():
                f.write(f"{model_name.upper()}\n")
                
                # Sort parameters by sensitivity
                param_sensitivity = []
                for param, results in params.items():
                    param_sensitivity.append((param, abs(results.get('sensitivity', 0))))
                
                sorted_params = sorted(param_sensitivity, key=lambda x: x[1], reverse=True)
                
                f.write("Parameters by Sensitivity (Most to Least):\n")
                for param, sensitivity in sorted_params:
                    f.write(f"  {param}: {sensitivity:.4f}\n")
                
                f.write("\n")
    
    print(f"Summary reports saved to {summary_dir} directory")

    # Save all models separately
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    for model_name, model in models.items():
        # Save model file
        model_filename = f"{models_dir}/{model_name}_model.pkl"
        joblib.dump(model, model_filename)
        
        # Save metadata
        model_metadata = results.get(model_name)
        if model_metadata:
            metadata_to_save = {
                "algorithm": model_metadata.get('algorithm', 'Unknown'),
                "metrics": model_metadata.get('metrics', {}),
                "hyperparameters": model_metadata.get('parameters', {})
            }
            metadata_filename = f"{models_dir}/{model_name}_metadata.json"
            with open(metadata_filename, 'w') as f:
                json.dump(metadata_to_save, f, indent=4)
        
    
    # Return the final results
    return {
        'integrated_data': engineered_data,
        'selected_features': selected_features,
        'models': models,
        'model_results': results,
        'optimal_params': optimal_params,
        'sensitivity_results': sensitivity_results,
        'prediction_system': prediction_system,
        'model_effectiveness': model_effectiveness 
    }


if __name__ == "__main__":
    # Run the pipeline with extended trials for production
    results = main(fast_mode=True)  # Use fast mode for development!
    #results = main(n_trials=100, n_iterations=200)
    
    print("\nModel training and evaluation completed successfully!")
    print("Models saved to 'models' directory")
    print("Visualizations saved to 'visualizations' directory")
    print("Summary reports saved to 'summary' directory")
    
    # Optionally, run a quick test of the prediction system
    try:
        # Load a sample input
        sample_input = {
            'temperature_mean': 25.0,
            'humidity_mean': 60.0,
            'rainfall_mean': 150.0,
            'ph_mean': 6.5,
            'nitrogen_mean': 80.0,
            'phosphorus_mean': 40.0,
            'potassium_mean': 50.0
        }
        
        print("\nRunning sample prediction with the following input:")
        for key, value in sample_input.items():
            print(f"  {key}: {value}")
        
        prediction = results['prediction_system'](sample_input)
        
        print("\nSample prediction results:")
        print(f"  Recommended crop: {prediction['recommended_crop']}")
        print(f"  Expected yield: {prediction['predictions'].get('yield', 'N/A')}")
        print(f"  Expected waste percentage: {prediction['predictions'].get('food_loss_pct', 'N/A')}%")
        print(f"  Effective yield: {prediction['predictions'].get('effective_yield', 'N/A')}")
        
        print("\nUse the trained model as follows:")
        print("  1. Import the module")
        print("  2. Call the main() function to train models")
        print("  3. Access the prediction_system from the returned results")
        print("  4. Pass your input parameters to the prediction system")
    except Exception as e:
        print(f"\nError during sample prediction: {e}")
        print("Model training and evaluation still completed successfully.")
