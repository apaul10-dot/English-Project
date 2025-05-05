# Required Libraries
import pandas as pd
from pathlib import Path
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
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
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

def clean_data(fao_data, crop_rec_data, crop_yield_data):
    """
    Perform initial data cleaning on all datasets with enhanced methods.
    """
    print("\nPerforming enhanced data cleaning...")
    
    # 2.1 Clean FAO data
    if fao_data is not None:
        print("Cleaning FAO data...")
        # Filter for India only
        fao_data.columns = fao_data.columns.str.strip()   # <-- add this
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
        crop_rec_data.columns = crop_rec_data.columns.str.strip()
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
        crop_yield_data.columns = crop_yield_data.columns.str.strip()  # <-- add this
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

def visualize_feature_distributions(original_data, transformed_data, dataset_name="dataset"):
    os.makedirs(f"visualizations/distributions/{dataset_name}", exist_ok=True)
    for col in original_data.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        sns.histplot(original_data[col].dropna(), kde=True)
        plt.title(f"Original: {col}")

        transformed_col = None
        for suffix in ["_log1p", "_cbrt", "_qt", "_scaled", "_yeojohnson", "_transformed"]:
            candidate = f"{col}{suffix}"
            if candidate in transformed_data.columns:
                transformed_col = candidate
                break

        if not transformed_col:
            continue

        if original_data[col].dropna().empty or transformed_data[transformed_col].dropna().empty:
            continue

        plt.subplot(1, 2, 2)
        sns.histplot(transformed_data[transformed_col].dropna(), kde=True, color='green')
        plt.title(f"Transformed: {transformed_col}")

        plt.tight_layout()
        plt.savefig(f"visualizations/distributions/{dataset_name}/{col}_vs_{transformed_col}.png")
        plt.close()

def transform_and_normalize_data(fao_data, crop_rec_data, crop_yield_data):
    print("\nPerforming advanced data transformation and normalization...")
    from sklearn.preprocessing import QuantileTransformer

    # --- FAO Data ---
    if fao_data is not None:
        print("Transforming FAO data...")
        fao_transformed = fao_data.copy()
        if 'loss_percentage' in fao_transformed.columns:
            scaler = RobustScaler()
            fao_transformed['loss_percentage_scaled'] = scaler.fit_transform(fao_transformed[['loss_percentage']])

        categorical_cols = [col for col in ['food_supply_stage', 'activity'] if col in fao_transformed.columns]
        for col in categorical_cols:
            if not fao_transformed[col].isna().all():
                value_counts = fao_transformed[col].value_counts()
                rare_categories = value_counts[value_counts < 3].index.tolist()
                if rare_categories:
                    fao_transformed[col] = fao_transformed[col].apply(lambda x: 'Other' if x in rare_categories else x)

                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded_data = encoder.fit_transform(fao_transformed[[col]].fillna('Missing'))
                encoded_df = pd.DataFrame(encoded_data,
                                          columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]])
                fao_transformed = pd.concat([fao_transformed.reset_index(drop=True),
                                             encoded_df.reset_index(drop=True)], axis=1)
    else:
        fao_transformed = None

    # --- Crop Recommendation Data ---
    if crop_rec_data is not None:
        print("Transforming crop recommendation data...")
        crop_rec_transformed = crop_rec_data.copy()
        numeric_cols = [col for col in ['nitrogen', 'phosphorus', 'potassium', 'temperature',
                                        'humidity', 'ph', 'rainfall'] if col in crop_rec_transformed.columns]

        for col in numeric_cols:
            values = crop_rec_transformed[col]
            if values.nunique() <= 1:
                continue

            skew = values.skew()
            transformed_candidates = {}

            try:
                if (values > 0).all():
                    log1p = np.log1p(values)
                    transformed_candidates['log1p'] = (log1p, pd.Series(log1p).skew())

                cbrt = np.cbrt(values)
                transformed_candidates['cbrt'] = (cbrt, pd.Series(cbrt).skew())

                qt = QuantileTransformer(output_distribution='normal', random_state=42)
                qt_trans = qt.fit_transform(values.values.reshape(-1, 1)).flatten()
                transformed_candidates['qt'] = (qt_trans, pd.Series(qt_trans).skew())

                transformed_candidates['original'] = (values, skew)

                best_key = min(transformed_candidates, key=lambda k: abs(transformed_candidates[k][1]))
                best_data, best_skew = transformed_candidates[best_key]

                suffix_map = {'log1p': '_log1p', 'cbrt': '_cbrt', 'qt': '_qt', 'original': '_scaled'}
                col_name = f"{col}{suffix_map[best_key]}"
                crop_rec_transformed[col_name] = best_data

                print(f"✅ {col} → {col_name} | skew before: {skew:.2f}, after: {best_skew:.2f}")
            except Exception as e:
                print(f"❌ {col} transformation failed: {e}")
    else:
        crop_rec_transformed = None

    # --- Crop Yield Data ---
    if crop_yield_data is not None:
        print("Transforming crop yield data...")
        crop_yield_transformed = crop_yield_data.copy()

        numeric_cols = crop_yield_transformed.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col in ['year', 'crop_year', 'area']:  # optionally skip stable IDs
                continue
            values = crop_yield_transformed[col]
            if values.nunique() <= 1:
                continue

            skew = values.skew()
            transformed_candidates = {}

            try:
                if (values > 0).all():
                    log1p = np.log1p(values)
                    transformed_candidates['log1p'] = (log1p, pd.Series(log1p).skew())

                cbrt = np.cbrt(values)
                transformed_candidates['cbrt'] = (cbrt, pd.Series(cbrt).skew())

                qt = QuantileTransformer(output_distribution='normal', random_state=42)
                qt_trans = qt.fit_transform(values.values.reshape(-1, 1)).flatten()
                transformed_candidates['qt'] = (qt_trans, pd.Series(qt_trans).skew())

                transformed_candidates['original'] = (values, skew)

                best_key = min(transformed_candidates, key=lambda k: abs(transformed_candidates[k][1]))
                best_data, best_skew = transformed_candidates[best_key]

                suffix_map = {'log1p': '_log1p', 'cbrt': '_cbrt', 'qt': '_qt', 'original': '_scaled'}
                col_name = f"{col}{suffix_map[best_key]}"
                crop_yield_transformed[col_name] = best_data

                print(f"✅ {col} → {col_name} | skew before: {skew:.2f}, after: {best_skew:.2f}")
            except Exception as e:
                print(f"❌ {col} transformation failed: {e}")

        # One-hot encode relevant categorical features
        categorical_cols = [col for col in ['state', 'season', 'crop'] if col in crop_yield_transformed.columns]
        for col in categorical_cols:
            value_counts = crop_yield_transformed[col].value_counts()
            rare_categories = value_counts[value_counts < 3].index.tolist()
            if rare_categories:
                crop_yield_transformed[col] = crop_yield_transformed[col].apply(lambda x: 'Other' if x in rare_categories else x)

            if col != 'crop':
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded_data = encoder.fit_transform(crop_yield_transformed[[col]].fillna('Missing'))
                encoded_df = pd.DataFrame(encoded_data,
                                          columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]])
                crop_yield_transformed = pd.concat([crop_yield_transformed.reset_index(drop=True),
                                                    encoded_df.reset_index(drop=True)], axis=1)
    else:
        crop_yield_transformed = None

    visualize_feature_distributions(crop_rec_data, crop_rec_transformed, dataset_name="crop_rec")
    visualize_feature_distributions(crop_yield_data, crop_yield_transformed, dataset_name="crop_yield")

    return fao_transformed, crop_rec_transformed, crop_yield_transformed

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

    # Ensure production, area, and yield are consistently handled
    if 'production' in data.columns and 'area' in data.columns:
        data['yield'] = data['production'] / data['area']
        print("✅ Recalculated 'yield' from 'production' and 'area'")

    if 'production' in data.columns and 'log1p_production' not in data.columns:
        data['log1p_production'] = np.log1p(data['production'])

    # Force-inclusion of transformed parameters
    key_transformed = ['fertilizer', 'pesticide', 'area']
    for param in key_transformed:
        if param in data.columns:
            data[f'log1p_{param}'] = np.log1p(data[param])
            print(f"✅ Applied log1p to '{param}'")

    # Ensure transformed features are preserved for selection
    required_log1p_features = ['log1p_fertilizer', 'log1p_pesticide', 'log1p_area']
    for feat in required_log1p_features:
        if feat not in data.columns:
            print(f"❌ Missing transformed feature: {feat} — check raw column existence.")
    
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
    
    # Identify target columns for unified model - both yield and waste
    target_cols = ['yield', 'post_harvest_waste_pct']
    available_targets = [col for col in target_cols if col in data.columns]
    
    if not available_targets:
        print("No target variables found. Cannot perform feature selection.")
        return data, data.columns.tolist()
    
    # Remove target columns from feature set
    exclude_cols += available_targets
    
    # Also remove derived targets
    derived_cols = ['effective_yield', 'waste_per_area']
    exclude_cols += [col for col in derived_cols if col in data.columns]
    
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
    
    # 5.4 Advanced feature selection using multiple methods
    # We'll combine methods to select features relevant for both yield and waste
    
    print("Performing multi-method feature selection...")
    
    # For each target, identify important features
    target_important_features = {}
    
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
            target_important_features[target] = top_features
    
    # 5.5 Create combined feature set for the unified model
    all_selected_features = set()
    for target, features in target_important_features.items():
        all_selected_features.update(features)
    
    # Add key agricultural parameters that might be important for control
    control_params = ['nitrogen_mean', 'phosphorus_mean', 'potassium_mean', 
                     'ph_mean', 'rainfall_mean', 'temperature_mean', 'humidity_mean']
    
    for param in control_params:
        if param in X_cols:
            all_selected_features.add(param)
    
    # Add derived features that might be important
    important_derived = ['npk_balance', 'soil_fertility', 'temp_humidity_index',
                        'aridity_index', 'ph_optimality']
    
    for feature in important_derived:
        if feature in data.columns:
            all_selected_features.add(feature)
    
    # Add crop and season indicators if available
    if 'crop_category' in data.columns:
        all_selected_features.add('crop_category')
    
    for col in data.columns:
        if col.startswith('crop_cat_') or col.startswith('is_'):
            all_selected_features.add(col)
    
    final_feature_list = list(all_selected_features)
    print(f"\nFinal combined feature set for unified model: {len(final_feature_list)} features")
    print(f"Sample features: {final_feature_list[:10]}...")


    os.makedirs("visualizations", exist_ok=True)
    plt.figure(figsize=(16, 12))
    # Force inclusion of key transformed features for visualization
    log1p_features = ['log1p_fertilizer', 'log1p_pesticide', 'log1p_area']
    for feature in log1p_features:
        if feature in data.columns and data[feature].dtype.kind in 'iufc':
            pass  # Keep it
        elif feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')  # Coerce non-numeric
        else:
            print(f"⚠️ Warning: {feature} not found in data, skipping inclusion in heatmap.")

    # Plot correlation heatmap including log1p features
    heatmap_cols = [col for col in data.columns if data[col].dtype.kind in 'iufc']
    print(heatmap_cols)
    corr_matrix = data[heatmap_cols].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap of Features (including log1p)")
    plt.tight_layout()
    plt.savefig("visualizations/correlation_heatmap.png")
    plt.close()
    
    return data, final_feature_list

# Define a unified agricultural model class that combines yield and waste prediction
class UnifiedAgriculturalModel(BaseEstimator, RegressorMixin):
    """
    A unified model that predicts both crop yield and post-harvest waste percentage.
    This allows for joint optimization of agricultural parameters to maximize effective yield.
    """
    def __init__(self, base_estimator='random_forest', **kwargs):
        self.base_estimator = base_estimator
        self.kwargs = kwargs
        self.yield_model = None
        self.waste_model = None
        self.crop_encoder = None
        self.feature_names = None
        
    def _get_estimator(self):
        """Create the specified base estimator with given parameters."""
        if self.base_estimator == 'random_forest':
            return RandomForestRegressor(**self.kwargs)
        elif self.base_estimator == 'gradient_boosting':
            return GradientBoostingRegressor(**self.kwargs)
        elif self.base_estimator == 'xgboost':
            return xgb.XGBRegressor(**self.kwargs)
        elif self.base_estimator == 'lightgbm':
            return lgb.LGBMRegressor(**self.kwargs)
        else:
            raise ValueError(f"Unsupported estimator: {self.base_estimator}")
            
    def fit(self, X, y):
        """
        Fit the model to predict both yield and waste.
        
        Args:
            X: Feature matrix
            y: Target matrix with columns for yield and waste
        """
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
            print("FEATURE NAMES:")
            print(self.feature_names)

            if 'crop' in X.columns:
                self.crop_encoder = LabelEncoder()
                X = X.copy()
                X['crop'] = self.crop_encoder.fit_transform(X['crop'])

            for col in X.select_dtypes(include='object').columns:
                X[col] = X[col].astype('category').cat.codes

        self.yield_model = self._get_estimator()  # ✅ Initialize yield_model
        self.yield_model.fit(X, y.iloc[:, 0])

        self.waste_model = self._get_estimator()
        self.waste_model.fit(X, y.iloc[:, 1])

        self.feature_names = X.columns.tolist()

        return self
    
    def predict(self, X):
        """
        Predict yield and waste for given inputs.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array with predictions for yield and waste
        """
        X = X.copy()

        # Ensure prediction data matches training feature set
        if hasattr(self, 'feature_names'):
            X = X[self.feature_names]  # Enforce same feature columns as in training

        # Handle crop encoding if needed
        if self.crop_encoder is not None and hasattr(X, 'columns') and 'crop' in X.columns:
            try:
                X['crop'] = self.crop_encoder.transform(X['crop'])
            except:
                unknown_mask = ~X['crop'].isin(self.crop_encoder.classes_)
                if unknown_mask.any():
                    X.loc[unknown_mask, 'crop'] = -1
                known_mask = ~unknown_mask
                if known_mask.any():
                    X.loc[known_mask, 'crop'] = self.crop_encoder.transform(X.loc[known_mask, 'crop'])

        # Encode any remaining object columns (e.g., 'Other', strings)
        if hasattr(X, 'select_dtypes'):
            for col in X.select_dtypes(include='object').columns:
                X[col] = X[col].astype('category').cat.codes

        # Make predictions
        yield_pred = self.yield_model.predict(X)
        waste_pred = self.waste_model.predict(X)

        waste_pred = np.clip(waste_pred, 0, 100)

        return np.column_stack((yield_pred, waste_pred))

    
    def get_feature_importance(self, target_idx=None):
        """
        Get feature importance for yield, waste, or combined.
        
        Args:
            target_idx: 0 for yield, 1 for waste, None for combined
            
        Returns:
            Array or dict of feature importances
        """
        if target_idx == 0:
            return self.yield_model.feature_importances_
        elif target_idx == 1:
            return self.waste_model.feature_importances_
        else:
            # Return combined importance (average of both models)
            combined = {
                'yield': self.yield_model.feature_importances_,
                'waste': self.waste_model.feature_importances_,
                'combined': (self.yield_model.feature_importances_ + 
                             self.waste_model.feature_importances_) / 2
            }
            return combined
        
    def calculate_effective_yield(self, X):
        """
        Calculate effective yield (yield after waste losses).
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of effective yield values
        """
        predictions = self.predict(X)
        yield_pred = predictions[:, 0]
        waste_pred = predictions[:, 1]
        
        # Calculate effective yield: yield * (1 - waste_percentage/100)
        effective_yield = yield_pred * (1 - waste_pred/100)
        return effective_yield
    
def develop_unified_model(data, feature_list, n_trials=50):
    """
    Develop a unified ML model for agricultural optimization that jointly 
    predicts crop yield and food loss with extensive hyperparameter optimization.
    
    Args:
        data: The integrated and feature-engineered dataset
        feature_list: List of selected features to use
        n_trials: Number of hyperparameter optimization trials
        
    Returns:
        Trained unified model and evaluation metrics
    """
    print("\nDeveloping unified agricultural optimization model...")
    
    if data is None or feature_list is None or len(feature_list) == 0:
        print("Insufficient data or features for model development.")
        return None, None, None
    
    # Check if we have the necessary target columns
    #required_targets = ['yield', 'post_harvest_waste_pct']
    required_targets = ['log1p_production', 'post_harvest_waste_pct']
    
    if not all(target in data.columns for target in required_targets):
        print("Missing required target columns. Need both 'yield' and 'post_harvest_waste_pct'.")
        missing = [target for target in required_targets if target not in data.columns]
        print(f"Missing targets: {missing}")
        return None, None, None
    
    # Use only available features from feature_list
    available_features = [feat for feat in feature_list if feat in data.columns]
    
    if len(available_features) < 2:
        print("Not enough features available for modeling.")
        return None, None, None
    
    print(f"Using {len(available_features)} features for unified model:")
    print(f"Sample features: {available_features[:5]}...")
    
    # Prepare the data
    # Remove rows with missing target values
    target_data = data[required_targets].copy()
    valid_rows = ~target_data.isna().any(axis=1)
    
    if valid_rows.sum() < 100:
        print(f"Not enough complete data rows for unified model. Found only {valid_rows.sum()} complete rows.")
        return None, None, None
    
    # Extract features and targets for valid rows
    X = data.loc[valid_rows, available_features].copy()
    y = data.loc[valid_rows, required_targets].copy()
    
    print(f"Proceeding with {len(X)} complete samples for unified model.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED
    )
    
    # Hyperparameter optimization with Optuna
    print(f"Optimizing hyperparameters for unified model with {n_trials} trials...")
    
    def objective(trial):
        # Try different algorithms
        estimator = trial.suggest_categorical('estimator', [
            'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'
        ])
        
        # Parameters depend on estimator type
        if estimator == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': RANDOM_SEED
            }
        elif estimator == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': RANDOM_SEED
            }
        elif estimator == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'random_state': RANDOM_SEED
            }
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
        
        # Create the unified model
        model = UnifiedAgriculturalModel(base_estimator=estimator, **params)
        
        try:
            # Use k-fold cross-validation to evaluate
            kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            scores = []
            
            for train_idx, val_idx in kf.split(X_train):
                # Split the folds
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                print(X_fold_train.columns)
                print(X_fold_val.columns)
                
                # Fit model on training fold
                model.fit(X_fold_train, y_fold_train)
            
                # Make predictions on validation fold
                y_pred = model.predict(X_fold_val)
                
                # Calculate metrics for each target
                r2_yield = r2_score(np.expm1(y_fold_val.iloc[:, 0]), np.expm1(y_pred[:, 0]))
                r2_waste = r2_score(y_fold_val.iloc[:, 1], y_pred[:, 1])
                
                # Calculate effective yield on validation set
                production_val = np.expm1(y_fold_val.iloc[:, 0])
                production_pred = np.expm1(y_pred[:, 0])
                area_val = X_fold_val['area'].replace(0, np.nan).values if 'area' in X_fold_val.columns else np.ones(len(production_val))
                yield_val = production_val / area_val
                yield_pred = production_pred / area_val
                waste_val = y_fold_val.iloc[:, 1]
                waste_pred = y_pred[:, 1]
                effective_yield_val = yield_val * (1 - waste_val / 100)
                effective_yield_pred = yield_pred * (1 - waste_pred / 100)

                r2_effective = r2_score(effective_yield_val, effective_yield_pred)
                
                # Combined score (weighted average of all three)
                score = (0.4 * r2_yield + 0.3 * r2_waste + 0.3 * r2_effective)
                scores.append(score)

            # Return mean score across folds
            return np.mean(scores)
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            return -1.0  # Penalty for errors
    
    # Create and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best parameters
    best_params = study.best_params
    best_estimator = best_params.pop('estimator')
    
    print(f"Best algorithm: {best_estimator}")
    print(f"Best parameters: {best_params}")
    
    # Train the final model with the best parameters
    final_model = UnifiedAgriculturalModel(base_estimator=best_estimator, **best_params)
    
    # Fit on the full training set
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    
    # Calculate effective yield metrics
    #effective_yield_test = y_test.iloc[:, 0] * (1 - y_test.iloc[:, 1]/100)
    if 'area' in X_test.columns:
        area_vals = X_test['area'].replace(0, np.nan).values
    else:
        area_vals = np.ones(len(y_test))

    production_test = np.expm1(y_test.iloc[:, 0].values)
    yield_test = production_test / area_vals
    waste_test = y_test.iloc[:, 1].values
    effective_yield_test = yield_test * (1 - waste_test / 100)

    # Ensure predictions also align
    production_pred = np.expm1(y_pred[:, 0])  # ✅ undo log1p
    waste_pred = y_pred[:, 1]
    yield_pred = production_pred / area_vals
    #effective_yield_pred = y_pred[:, 0] * (1 - y_pred[:, 1]/100)
    effective_yield_pred = yield_pred * (1 - waste_pred / 100)

    # Calculate metrics
    r2_yield = r2_score(production_test, production_pred)
    rmse_yield = np.sqrt(mean_squared_error(production_test, production_pred))
    mae_yield = mean_absolute_error(production_test, production_pred)

    r2_waste = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
    rmse_waste = np.sqrt(mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]))
    mae_waste = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
    
    r2_effective = r2_score(effective_yield_test, effective_yield_pred)
    rmse_effective = np.sqrt(mean_squared_error(effective_yield_test, effective_yield_pred))
    
    print("\nUnified Model Performance Metrics:")
    print("\nYield Prediction:")
    print(f"R² Score: {r2_yield:.4f}")
    print(f"RMSE: {rmse_yield:.4f}")
    print(f"MAE: {mae_yield:.4f}")
    
    print("\nWaste Percentage Prediction:")
    print(f"R² Score: {r2_waste:.4f}")
    print(f"RMSE: {rmse_waste:.4f}")
    print(f"MAE: {mae_waste:.4f}")
    
    print("\nEffective Yield Prediction:")
    print(f"R² Score: {r2_effective:.4f}")
    print(f"RMSE: {rmse_effective:.4f}")
    
    # Store results
    results = {
        'algorithm': best_estimator,
        'parameters': best_params,
        'metrics': {
            'yield': {
                'r2_score': r2_yield,
                'rmse': rmse_yield,
                'mae': mae_yield
            },
            'waste': {
                'r2_score': r2_waste,
                'rmse': rmse_waste,
                'mae': mae_waste
            },
            'effective_yield': {
                'r2_score': r2_effective,
                'rmse': rmse_effective
            }
        }
    }
    
    # Save test set for later evaluation
    test_sets = {
        "X_test": X_test,
        "y_test": y_test
    }
    
    # Create visualization of feature importance
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get feature importances
    importances = final_model.get_feature_importance()
    
    for target_name, importance_values in [
        ('yield', importances['yield']), 
        ('waste', importances['waste']),
        ('combined', importances['combined'])
    ]:
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': available_features,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title(f'Top 20 Feature Importance for {target_name.title()} Prediction')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/unified_model_{target_name}_importance.png")
        plt.close()

    # Save baseline predictions on test set for comparison
    baseline_predictions = final_model.predict(X_test)
    #test_sets["baseline_predictions"] = baseline_predictions
    test_sets['baseline_yield'] = yield_pred
    
    return final_model, results, test_sets

def optimize_agricultural_parameters(model, test_sets, data, available_features, n_iterations=100):
    """
    Optimize controllable agricultural parameters to maximize yield and minimize waste.
    Applies inverse transformations and ensures feature alignment.
    """
    print("\nPerforming unified agricultural parameter optimization...")

    if model is None:
        print("No model available for optimization. Skipping this step.")
        return None

    fixed_keywords = ['state', 'region', 'country', 'rainfall', 'temperature', 'humidity', 'aridity', 'ph', 'year',
                      'nitrogen', 'phosphorus', 'potassium', 'npk', 'p_ratio']
    base_controllable_params = ['crop', 'season', 'area', 'fertilizer', 'pesticide']

    print("\nAvailable feature columns (grouped):")
    print("→ Controllable candidates:")
    for col in data.columns:
        if any(k in col for k in base_controllable_params):
            print(f"  {col}")
    print("\n→ Fixed/environmental candidates:")
    for col in data.columns:
        if any(k in col for k in fixed_keywords):
            print(f"  {col}")

    controllable_params = [f for f in available_features if any(k in f for k in base_controllable_params)]
    controllable_params = list(set(controllable_params + base_controllable_params))
    controllable_params = [p for p in controllable_params if p in data.columns and not any(k in p for k in fixed_keywords)]

    param_bounds = {}
    for param in controllable_params:
        if param == 'crop':
            continue
        elif data[param].dtype.kind in 'iufc':
            min_val = data[param].min()
            max_val = data[param].max()
            range_val = max_val - min_val
            param_bounds[param] = (max(0, min_val - 0.3 * range_val), max_val + 0.3 * range_val)
        else:
            print(f"⚠️ Skipping non-numeric param: {param} (dtype={data[param].dtype})")

    print(controllable_params)

    print(f"\nOptimizing {len(param_bounds)} numeric controllable parameters:")
    for param in param_bounds:
        print(f"  {param}: {param_bounds[param]}")

    scenarios = []
    if 'crop' in data.columns:
        top_crops = data['crop'].value_counts().head(3).index.tolist()
        for crop in top_crops:
            scenarios.append({
                'name': f"optimize_for_{crop}",
                'description': f"Optimize for fixed crop: {crop}",
                'fixed_params': {'crop': crop},
                'objective': 'effective_yield'
            })

    scenarios.extend([
        {'name': "maximize_effective_yield", 'description': "Maximize effective yield (free crop)", 'fixed_params': {}, 'objective': 'effective_yield'},
        {'name': "maximize_raw_yield", 'description': "Maximize raw yield", 'fixed_params': {}, 'objective': 'yield'},
        {'name': "minimize_waste", 'description': "Minimize food waste %", 'fixed_params': {}, 'objective': 'waste'}
    ])

    X_baseline = test_sets["X_test"].iloc[[0]].copy()
    base_pred = model.predict(X_baseline)[0]
    #base_yield = np.expm1(base_pred[0])
    base_production = np.expm1(base_pred[0])  # Undo log1p
    base_area = X_baseline['area'].values[0] if 'area' in X_baseline.columns else 1.0
    base_yield = base_production / base_area
    base_waste = base_pred[1]
    base_effective = base_yield * (1 - base_waste / 100)

    optimization_results = {}

    for scenario in scenarios:
        print(f"\nRunning optimization for scenario: {scenario['name']}")
        opt_params = {p: param_bounds[p] for p in param_bounds if p not in scenario['fixed_params']}
        print(f"  Fixed parameters: {scenario['fixed_params']}")
        print(f"  Optimizable parameters: {opt_params}")

        def objective_fn(**params):
            # Start from a clean copy
            sample = test_sets["X_test"].iloc[[0]].copy()

            # Apply fixed parameters
            for k, v in scenario['fixed_params'].items():
                sample[k] = v

            # Define which columns are log-transformed
            log_transform_cols = ['fertilizer', 'pesticide', 'area']

            # Apply parameter updates
            for k, v in params.items():
                if k in log_transform_cols:
                    sample[f'log1p_{k}'] = np.log1p(v)
                else:
                    sample[k] = v

            print(f"model feature name {model.feature_names}")
            # ✅ Only keep training features
            sample = sample[model.feature_names]  # Use saved list from training

            try:
                # ✅ Clean: just enforce training-time features
                sample = sample[model.feature_names]

                pred = model.predict(sample)[0]
                prod = np.expm1(pred[0])

                # ✅ Correct handling of area (transformed or not)
                if 'log1p_area' in sample.columns:
                    area_val = np.expm1(sample['log1p_area'].values[0])
                elif 'area' in sample.columns:
                    area_val = sample['area'].values[0]
                else:
                    area_val = 1.0
                    print("⚠️ Warning: area not found in sample, using default value of 1.0")

                yield_val = prod / area_val
                waste_val = np.clip(pred[1], 0, 100)
                eff = yield_val * (1 - waste_val / 100)

                if scenario['objective'] == 'yield':
                    return yield_val - 0.5 * waste_val
                elif scenario['objective'] == 'waste':
                    return -waste_val + 0.2 * yield_val
                elif scenario['objective'] == 'effective_yield':
                    return eff - 0.5 * waste_val
            except Exception as e:
                print(f"Prediction error: {e}")
                return -999


        # 🟢 Now safe to call optimizer
        optimizer = BayesianOptimization(f=objective_fn, pbounds=opt_params, random_state=42, verbose=2)
        optimizer.maximize(init_points=5, n_iter=n_iterations)

        # 🟢 This is AFTER the optimizer has run
        best = optimizer.max['params']
        for k, v in scenario['fixed_params'].items():
            best[k] = v


        sample = test_sets["X_test"].iloc[[0]].copy()
        for k, v in best.items():
            if f'log1p_{k}' in sample.columns:
                sample[f'log1p_{k}'] = np.log1p(v)
            elif k in sample.columns:
                sample[k] = v
        
        print("Done with Bayesian Optimization. Checkign for final...")


        pred = model.predict(sample)[0]
        #y = np.expm1(pred[0])
        prod = np.expm1(pred[0])  # Undo log1p  # predicted production (already modeled)
        # ✅ Correct handling of area (transformed or not)
        if 'area' in sample.columns:
            area_val = sample['area'].values[0]
        elif 'log1p_area' in sample.columns:
            area_val = np.expm1(sample['log1p_area'].values[0])
        else:
            area_val = 1.0
            print("⚠️ Warning: area not found in sample, using default value of 1.0")

        y = prod / area_val  # compute yield
        w = np.clip(pred[1], 0, 100)  # clamp waste to [0, 100]%
        eff = y * (1 - w / 100)  # effective yield
        #eff = yield_val * (1 - w / 100)    --- was never told to put here, but just in case, this might be the answer


        print(f"Optimized results — Yield: {y:.2f}, Waste: {w:.2f}%, Effective Yield: {eff:.2f}")

        optimization_results[scenario['name']] = {
            'parameters': best,
            'expected_yield': y,
            'expected_loss': w,
            'effective_yield': eff,
            'improvement': {
                'yield_gain': y - base_yield,
                'waste_reduction': base_waste - w,
                'effective_yield_gain': eff - base_effective
            }
        }
    
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
        plt.savefig("visualizations/unified_optimization_comparison.png")
        plt.close()


        print("\nCreating baseline vs optimized comparison visualizations...")

        for scenario, result in optimization_results.items():
            improvement = result.get("improvement", {})
            if not improvement:
                continue

            # Labels and values
            metrics = ['Yield', 'Waste %', 'Effective Yield']
            baseline_vals = [base_yield, base_waste, base_effective]
            optimized_vals = [result['expected_yield'], result['expected_loss'], result['effective_yield']]

            # Plot
            plt.figure(figsize=(10, 6))
            bar_width = 0.35
            r1 = np.arange(len(metrics))
            r2 = [x + bar_width for x in r1]

            plt.bar(r1, baseline_vals, width=bar_width, label='Baseline')
            plt.bar(r2, optimized_vals, width=bar_width, label='Optimized')

            plt.xticks([r + bar_width / 2 for r in r1], metrics)
            plt.ylabel('Value')
            plt.title(f'Baseline vs Optimized Values ({scenario.replace("_", " ").title()})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"visualizations/baseline_vs_optimized_{scenario}.png")
            plt.close()


        # Plot improvements separately
        plt.figure(figsize=(8, 5))
        improvement_vals = [
            improvement["yield_gain"],
            improvement["waste_reduction"],
            improvement["effective_yield_gain"]
        ]
        colors = ['green' if val >= 0 else 'red' for val in improvement_vals]
        plt.bar(metrics, improvement_vals, color=colors)
        plt.ylabel('Improvement')
        plt.title(f'Improvement Over Baseline ({scenario.replace("_", " ").title()})')
        plt.axhline(0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(f"visualizations/improvement_comparison_{scenario}.png")
        plt.close()
    
    return optimization_results

def evaluate_unified_model(model, test_sets, data):
    """
    Perform detailed evaluation of the unified model on test data
    with visualizations and analysis.
    
    Args:
        model: Trained unified model
        test_sets: Dictionary containing test data
        data: The complete dataset
        
    Returns:
        Dictionary of evaluation metrics and insights
    """
    
    print("\nPerforming detailed evaluation of unified model...")
    
    if model is None or test_sets is None:
        print("Model or test data not available. Skipping evaluation.")
        return None
    
    # Create visualizations directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Extract test data
    X_test = test_sets["X_test"]
    y_test = test_sets["y_test"]
    
    # Get predictions on test set
    y_pred = model.predict(X_test)
    
    # Step 1: Get area
    area_vals = X_test['area'].replace(0, np.nan).values if 'area' in X_test.columns else np.ones(len(X_test))

    # Step 2: Convert log1p(production) → production → yield
    production_true = np.expm1(y_test.iloc[:, 0].values)
    production_pred = np.expm1(y_pred[:, 0])

    yield_true = production_true / area_vals
    yield_pred = production_pred / area_vals

    waste_true = y_test.iloc[:, 1].values
    waste_pred = y_pred[:, 1]

    # Step 3: Build dataframe with corrected values
    pred_df = pd.DataFrame({
        'yield_true': yield_true,
        'yield_pred': yield_pred,
        'waste_true': waste_true,
        'waste_pred': waste_pred
    })
    
    # Calculate effective yield (yield after waste)
    pred_df['effective_yield_true'] = pred_df['yield_true'] * (1 - pred_df['waste_true']/100)
    pred_df['effective_yield_pred'] = pred_df['yield_pred'] * (1 - pred_df['waste_pred']/100)
    
    # 1. Basic error metrics
    metrics = {}
    
    # Yield metrics
    metrics['yield'] = {
        'r2': r2_score(pred_df['yield_true'], pred_df['yield_pred']),
        'rmse': np.sqrt(mean_squared_error(pred_df['yield_true'], pred_df['yield_pred'])),
        'mae': mean_absolute_error(pred_df['yield_true'], pred_df['yield_pred']),
        'mape': np.mean(np.abs((pred_df['yield_true'] - pred_df['yield_pred']) / 
                               (pred_df['yield_true'] + 1e-10))) * 100  # Add small value to avoid division by zero
    }
    
    # Waste metrics
    metrics['waste'] = {
        'r2': r2_score(pred_df['waste_true'], pred_df['waste_pred']),
        'rmse': np.sqrt(mean_squared_error(pred_df['waste_true'], pred_df['waste_pred'])),
        'mae': mean_absolute_error(pred_df['waste_true'], pred_df['waste_pred']),
        'mape': np.mean(np.abs((pred_df['waste_true'] - pred_df['waste_pred']) / 
                               (pred_df['waste_true'] + 1e-10))) * 100
    }
    
    # Effective yield metrics
    metrics['effective_yield'] = {
        'r2': r2_score(pred_df['effective_yield_true'], pred_df['effective_yield_pred']),
        'rmse': np.sqrt(mean_squared_error(pred_df['effective_yield_true'], pred_df['effective_yield_pred'])),
        'mae': mean_absolute_error(pred_df['effective_yield_true'], pred_df['effective_yield_pred']),
        'mape': np.mean(np.abs((pred_df['effective_yield_true'] - pred_df['effective_yield_pred']) / 
                               (pred_df['effective_yield_true'] + 1e-10))) * 100
    }
    
    # Print metrics
    print("\nUnified Model Evaluation Metrics:")
    for target, target_metrics in metrics.items():
        print(f"\n{target.replace('_', ' ').title()} Metrics:")
        for metric_name, value in target_metrics.items():
            print(f"  {metric_name.upper()}: {value:.4f}")
    
    # 2. Create prediction vs actual scatter plots
    plt.figure(figsize=(18, 6))
    
    # Yield scatter plot
    plt.subplot(1, 3, 1)
    plt.scatter(pred_df['yield_true'], pred_df['yield_pred'], alpha=0.6)
    plt.plot([pred_df['yield_true'].min(), pred_df['yield_true'].max()],
             [pred_df['yield_true'].min(), pred_df['yield_true'].max()],
             'r--', linewidth=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'Yield Prediction (R² = {metrics["yield"]["r2"]:.3f})')
    
    # Waste scatter plot
    plt.subplot(1, 3, 2)
    plt.scatter(pred_df['waste_true'], pred_df['waste_pred'], alpha=0.6)
    plt.plot([pred_df['waste_true'].min(), pred_df['waste_true'].max()],
             [pred_df['waste_true'].min(), pred_df['waste_true'].max()],
             'r--', linewidth=2)
    plt.xlabel('Actual Waste %')
    plt.ylabel('Predicted Waste %')
    plt.title(f'Waste Prediction (R² = {metrics["waste"]["r2"]:.3f})')
    
    # Effective yield scatter plot
    plt.subplot(1, 3, 3)
    plt.scatter(pred_df['effective_yield_true'], pred_df['effective_yield_pred'], alpha=0.6)
    plt.plot([pred_df['effective_yield_true'].min(), pred_df['effective_yield_true'].max()],
             [pred_df['effective_yield_true'].min(), pred_df['effective_yield_true'].max()],
             'r--', linewidth=2)
    plt.xlabel('Actual Effective Yield')
    plt.ylabel('Predicted Effective Yield')
    plt.title(f'Effective Yield Prediction (R² = {metrics["effective_yield"]["r2"]:.3f})')
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/unified_model_predictions.png")
    plt.close()
    
    # 3. Error analysis across different segments
    # If crop information is available, analyze errors by crop type
    error_analysis = {}
    
    if 'crop' in X_test.columns:
        # Add crop information to the prediction dataframe
        pred_df['crop'] = X_test['crop'].values
        
        # Group by crop and calculate mean errors
        crop_errors = pred_df.groupby('crop').apply(lambda g: pd.Series({
            'yield_mae': mean_absolute_error(g['yield_true'], g['yield_pred']),
            'waste_mae': mean_absolute_error(g['waste_true'], g['waste_pred']),
            'effective_yield_mae': mean_absolute_error(g['effective_yield_true'], g['effective_yield_pred']),
            'count': len(g)
        })).sort_values('count', ascending=False)
        
        # Only include crops with enough samples
        crop_errors = crop_errors[crop_errors['count'] >= 5]
        
        # Create error by crop visualization
        if len(crop_errors) > 1:
            plt.figure(figsize=(14, 8))
            
            # Set width of bars
            barWidth = 0.25
            
            # Set positions of the bars on X axis
            r1 = np.arange(len(crop_errors))
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            
            # Create bars
            plt.bar(r1, crop_errors['yield_mae'], width=barWidth, label='Yield MAE')
            plt.bar(r2, crop_errors['waste_mae'], width=barWidth, label='Waste MAE')
            plt.bar(r3, crop_errors['effective_yield_mae'], width=barWidth, label='Effective Yield MAE')
            
            # Add labels and legend
            plt.xlabel('Crop')
            plt.ylabel('Mean Absolute Error')
            plt.title('Model Error by Crop Type')
            plt.xticks([r + barWidth for r in range(len(crop_errors))], crop_errors.index, rotation=45)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/unified_model_error_by_crop.png")
            plt.close()
            
            error_analysis['error_by_crop'] = crop_errors.to_dict()
    
    # 4. Residual analysis
    # Calculate residuals
    pred_df['yield_residual'] = pred_df['yield_true'] - pred_df['yield_pred']
    pred_df['waste_residual'] = pred_df['waste_true'] - pred_df['waste_pred']
    pred_df['effective_yield_residual'] = pred_df['effective_yield_true'] - pred_df['effective_yield_pred']
    
    # Plot residual distributions
    plt.figure(figsize=(18, 6))
    
    # Yield residuals
    plt.subplot(1, 3, 1)
    sns.histplot(pred_df['yield_residual'], kde=True)
    plt.xlabel('Yield Residual')
    plt.ylabel('Frequency')
    plt.title('Yield Prediction Residuals')
    
    # Waste residuals
    plt.subplot(1, 3, 2)
    sns.histplot(pred_df['waste_residual'], kde=True)
    plt.xlabel('Waste % Residual')
    plt.ylabel('Frequency')
    plt.title('Waste Prediction Residuals')
    
    # Effective yield residuals
    plt.subplot(1, 3, 3)
    sns.histplot(pred_df['effective_yield_residual'], kde=True)
    plt.xlabel('Effective Yield Residual')
    plt.ylabel('Frequency')
    plt.title('Effective Yield Prediction Residuals')
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/unified_model_residuals.png")
    plt.close()
    
    # 5. Check if residuals correlate with any features
    residual_correlations = {}
    
    for col in X_test.columns:
        if pd.api.types.is_numeric_dtype(X_test[col]):
            yield_corr = np.corrcoef(X_test[col], pred_df['yield_residual'])[0, 1]
            waste_corr = np.corrcoef(X_test[col], pred_df['waste_residual'])[0, 1]
            
            if abs(yield_corr) > 0.3 or abs(waste_corr) > 0.3:
                residual_correlations[col] = {
                    'yield_correlation': yield_corr,
                    'waste_correlation': waste_corr
                }
    
    if residual_correlations:
        print("\nFeatures with strong correlation to residuals:")
        for feature, corrs in residual_correlations.items():
            print(f"  {feature}:")
            print(f"    Yield residual correlation: {corrs['yield_correlation']:.4f}")
            print(f"    Waste residual correlation: {corrs['waste_correlation']:.4f}")
    
    # Return all evaluation results
    evaluation_results = {
        'metrics': metrics,
        'error_analysis': error_analysis,
        'residual_correlations': residual_correlations
    }
    
    return evaluation_results


def perform_unified_sensitivity_analysis(model, test_sets, data):
    """
    Perform sensitivity analysis to determine how the model predictions
    change with variations in input parameters.
    
    Args:
        model: Trained unified model
        test_sets: Dictionary containing test data
        data: The complete dataset
        
    Returns:
        Tuple of (sensitivity_results, optimal_values)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    print("\nPerforming sensitivity analysis for the unified model...")
    
    if model is None or test_sets is None:
        print("Model or test data not available. Skipping sensitivity analysis.")
        return None, None
    
    # Create visualizations directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get a representative sample from the test set
    X_test = test_sets["X_test"]
    
    # Use the mean values as our base case
    # This represents an "average" scenario
    base_case = pd.DataFrame(X_test.mean(numeric_only=True)).T
    
    # Make sure we have all needed columns
    for col in X_test.columns:
        if col not in base_case.columns:
            if col in X_test and X_test[col].dtype == 'object':
                # For categorical features, use the most common value
                base_case[col] = X_test[col].mode()[0]
    
    # Parameters to analyze (agricultural inputs that can be controlled)
    params_to_analyze = [
        'nitrogen_mean', 'phosphorus_mean', 'potassium_mean',  # Fertilizer
        'ph_mean',  # Soil amendments
        'rainfall_mean',  # Irrigation
        'temperature_mean', 'humidity_mean'  # Environmental factors
    ]
    
    # Filter to params that exist in our data
    params_to_analyze = [p for p in params_to_analyze if p in base_case.columns]
    
    print(f"Analyzing sensitivity for {len(params_to_analyze)} key parameters:")
    for param in params_to_analyze:
        print(f"  - {param}")
    
    # Define range for each parameter
    # We'll use percentiles from the data for realistic ranges
    param_ranges = {}
    
    for param in params_to_analyze:
        # Use 5th to 95th percentile range
        min_val = X_test[param].quantile(0.05)
        max_val = X_test[param].quantile(0.95)
        
        # Create 20 evenly spaced values across the range
        param_ranges[param] = np.linspace(min_val, max_val, 20)
    
    # Store sensitivity results
    sensitivity_results = {}
    optimal_values = {}
    
    # For each parameter, vary it and observe changes in predictions
    for param in params_to_analyze:
        print(f"\nAnalyzing sensitivity to {param}...")
        
        param_values = param_ranges[param]
        yield_predictions = []
        waste_predictions = []
        effective_yield_predictions = []
        
        # For each value in the range
        for value in param_values:
            # Create a copy of the base case
            test_case = base_case.copy()
            
            # Set the parameter to the current value
            test_case[param] = value

            # Ensure test_case has exactly the same columns in same order as during training
            test_case = test_case.reindex(model.feature_names, axis=1)

            
            # Predict yield and waste
            predictions = model.predict(test_case)
            # Convert log1p(production) → production → yield
            production_pred = np.expm1(predictions[0, 0])
            area_val = test_case['area'].values[0] if 'area' in test_case.columns else 1.0
            predicted_yield = production_pred / area_val
            predicted_waste = predictions[0, 1]
            
            # Calculate effective yield
            effective_yield = predicted_yield * (1 - predicted_waste/100)
            
            # Store predictions
            yield_predictions.append(predicted_yield)
            waste_predictions.append(predicted_waste)
            effective_yield_predictions.append(effective_yield)
        
        # Plot relationship between parameter and predictions
        plt.figure(figsize=(12, 8))
        
        # Standardize values for easier comparison
        param_values_std = (param_values - np.mean(param_values)) / np.std(param_values)
        
        # Primary Y-axis for yield
        ax1 = plt.gca()
        line1 = ax1.plot(param_values, yield_predictions, 'b-', label='Yield')
        ax1.set_xlabel(param)
        ax1.set_ylabel('Yield', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Secondary Y-axis for waste percentage
        ax2 = ax1.twinx()
        line2 = ax2.plot(param_values, waste_predictions, 'r-', label='Waste %')
        ax2.set_ylabel('Waste %', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Third Y-axis for effective yield
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))  # Offset the right spine
        line3 = ax3.plot(param_values, effective_yield_predictions, 'g-', label='Effective Yield')
        ax3.set_ylabel('Effective Yield', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        
        # Add all lines to the legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc='best')
        
        plt.title(f'Effect of {param} on Yield, Waste, and Effective Yield')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/sensitivity_{param}.png")
        plt.close()
        
        # Calculate sensitivity metrics
        # Elasticity: % change in output / % change in input
        # Here we use linear regression slope as proxy for average elasticity
        param_arr = np.array(param_values).reshape(-1, 1)
        yield_elasticity = np.polyfit(param_values, yield_predictions, 1)[0] * np.mean(param_values) / np.mean(yield_predictions)
        waste_elasticity = np.polyfit(param_values, waste_predictions, 1)[0] * np.mean(param_values) / np.mean(waste_predictions)
        effective_elasticity = np.polyfit(param_values, effective_yield_predictions, 1)[0] * np.mean(param_values) / np.mean(effective_yield_predictions)
        
        # Find optimal value for effective yield
        effective_yield_array = np.array(effective_yield_predictions)
        optimal_idx = np.argmax(effective_yield_array)
        optimal_value = param_values[optimal_idx]
        
        # Store results
        sensitivity_results[param] = {
            'param_values': param_values.tolist(),
            'yield_predictions': yield_predictions,
            'waste_predictions': waste_predictions,
            'effective_yield_predictions': effective_yield_predictions,
            'yield_elasticity': yield_elasticity,
            'waste_elasticity': waste_elasticity,
            'effective_yield_elasticity': effective_elasticity,
            'optimal_value': optimal_value,
            'max_effective_yield': effective_yield_array[optimal_idx]
        }
        
        optimal_values[param] = optimal_value
        
        print(f"  Yield elasticity: {yield_elasticity:.4f}")
        print(f"  Waste elasticity: {waste_elasticity:.4f}")
        print(f"  Effective yield elasticity: {effective_elasticity:.4f}")
        print(f"  Optimal value for maximizing effective yield: {optimal_value:.4f}")
    
    # Create a summary plot of elasticities
    params = list(sensitivity_results.keys())
    yield_elasticities = [sensitivity_results[p]['yield_elasticity'] for p in params]
    waste_elasticities = [sensitivity_results[p]['waste_elasticity'] for p in params]
    effective_elasticities = [sensitivity_results[p]['effective_yield_elasticity'] for p in params]
    
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(params))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, yield_elasticities, width=barWidth, label='Yield Elasticity')
    plt.bar(r2, waste_elasticities, width=barWidth, label='Waste Elasticity')
    plt.bar(r3, effective_elasticities, width=barWidth, label='Effective Yield Elasticity')
    
    # Add labels and legend
    plt.xlabel('Parameter')
    plt.ylabel('Elasticity')
    plt.title('Parameter Sensitivity (Elasticity)')
    plt.xticks([r + barWidth for r in range(len(params))], params, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/parameter_elasticities.png")
    plt.close()
    
    print("\nParameter sensitivity analysis complete.")
    
    return sensitivity_results, optimal_values

def analyze_model_effectiveness(model, test_sets, optimization_results):
    """
    Analyze the model effectiveness by comparing optimized parameters with baseline
    and calculating potential improvements in yield and waste reduction.
    
    Args:
        model: Trained unified model
        test_sets: Dictionary containing test data
        optimization_results: Results from parameter optimization
        
    Returns:
        Dictionary of effectiveness metrics and insights
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    print("\nAnalyzing model effectiveness and potential benefits...")
    
    if model is None or test_sets is None or optimization_results is None:
        print("Model, test data, or optimization results not available. Skipping analysis.")
        return None
    
    # Create visualizations directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get test data
    X_test = test_sets["X_test"]
    y_test = test_sets["y_test"]
    
    # Calculate baseline performance (average test set values)
    baseline_yield = y_test.iloc[:, 0].mean()
    baseline_waste = y_test.iloc[:, 1].mean()
    baseline_effective = baseline_yield * (1 - baseline_waste/100)
    
    print(f"\nBaseline performance (test set averages):")
    print(f"  Average yield: {baseline_yield:.2f}")
    print(f"  Average waste: {baseline_waste:.2f}%")
    print(f"  Average effective yield: {baseline_effective:.2f}")
    
    # Analyze each optimization scenario
    scenario_improvements = {}
    
    for scenario_name, result in optimization_results.items():
        # Get optimized predictions
        opt_yield = result['expected_yield']
        opt_waste = result['expected_loss']
        opt_effective = result['effective_yield']
        
        # Calculate improvements
        yield_improvement = ((opt_yield / baseline_yield) - 1) * 100
        waste_reduction = baseline_waste - opt_waste
        effective_improvement = ((opt_effective / baseline_effective) - 1) * 100
        
        # Store results
        scenario_improvements[scenario_name] = {
            'yield_improvement_pct': yield_improvement,
            'waste_reduction_pts': waste_reduction,
            'effective_yield_improvement_pct': effective_improvement,
            'optimized_yield': opt_yield,
            'optimized_waste': opt_waste,
            'optimized_effective_yield': opt_effective
        }
        
        print(f"\nScenario: {scenario_name}")
        print(f"  Yield improvement: {yield_improvement:.2f}%")
        print(f"  Waste reduction: {waste_reduction:.2f} percentage points")
        print(f"  Effective yield improvement: {effective_improvement:.2f}%")
    
    # Create comparison visualization
    scenarios = list(scenario_improvements.keys())
    yield_improvements = [scenario_improvements[s]['yield_improvement_pct'] for s in scenarios]
    waste_reductions = [scenario_improvements[s]['waste_reduction_pts'] for s in scenarios]
    effective_improvements = [scenario_improvements[s]['effective_yield_improvement_pct'] for s in scenarios]
    
    plt.figure(figsize=(14, 8))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(scenarios))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, yield_improvements, width=barWidth, label='Yield Improvement (%)')
    plt.bar(r2, waste_reductions, width=barWidth, label='Waste Reduction (percentage points)')
    plt.bar(r3, effective_improvements, width=barWidth, label='Effective Yield Improvement (%)')
    
    # Add labels and legend
    plt.xlabel('Optimization Scenario')
    plt.ylabel('Improvement')
    plt.title('Potential Improvements from Model Optimization')
    plt.xticks([r + barWidth for r in range(len(scenarios))], [s.replace('_', ' ').title() for s in scenarios], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/model_effectiveness.png")
    plt.close()
    
    # Calculate economic potential
    # Assume a simple economic model based on average crop values
    # This is a placeholder - in a real implementation, you would use crop-specific price data
    
    # Sample placeholder crop values ($/unit)
    crop_values = {
        'Rice': 2.5,
        'Wheat': 2.0,
        'Maize': 1.8,
        'Potato': 1.2,
        'Sugarcane': 0.5,
        'default': 1.5  # Default value for crops not listed
    }
    
    # Calculate economic impact for scenarios that include crop information
    economic_impact = {}
    
    for scenario_name, result in optimization_results.items():
        # Check if there's a specific crop for this scenario
        crop = None
        if 'parameters' in result and 'crop' in result['parameters']:
            crop = result['parameters']['crop']
        elif 'fixed_params' in result and 'crop' in result['fixed_params']:
            crop = result['fixed_params']['crop']
        
        # Get crop value
        crop_value = crop_values.get(crop, crop_values['default']) if crop else crop_values['default']
        
        # Calculate economic impact based on effective yield improvement
        baseline_value = baseline_effective * crop_value
        optimized_value = result['effective_yield'] * crop_value
        value_improvement = optimized_value - baseline_value
        
        # Store results
        economic_impact[scenario_name] = {
            'crop': crop,
            'crop_value': crop_value,
            'baseline_value': baseline_value,
            'optimized_value': optimized_value,
            'value_improvement': value_improvement,
            'value_improvement_pct': (value_improvement / baseline_value) * 100 if baseline_value > 0 else 0
        }
        
        print(f"\nEconomic Analysis - {scenario_name}:")
        print(f"  Crop: {crop if crop else 'Not specified'}")
        print(f"  Baseline value: ${baseline_value:.2f}")
        print(f"  Optimized value: ${optimized_value:.2f}")
        print(f"  Value improvement: ${value_improvement:.2f} ({economic_impact[scenario_name]['value_improvement_pct']:.2f}%)")
    
    # Create economic impact visualization
    if economic_impact:
        scenarios = list(economic_impact.keys())
        value_improvements = [economic_impact[s]['value_improvement'] for s in scenarios]
        value_pct_improvements = [economic_impact[s]['value_improvement_pct'] for s in scenarios]
        
        plt.figure(figsize=(14, 6))
        
        # Primary Y-axis for absolute improvement
        ax1 = plt.gca()
        ax1.bar(scenarios, value_improvements, color='green', alpha=0.7)
        ax1.set_xlabel('Optimization Scenario')
        ax1.set_ylabel('Value Improvement ($)', color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        plt.xticks(rotation=45, ha='right')
        
        # Secondary Y-axis for percentage improvement
        ax2 = ax1.twinx()
        ax2.plot(scenarios, value_pct_improvements, 'ro-', linewidth=2)
        ax2.set_ylabel('Value Improvement (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Economic Impact of Optimization')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/economic_impact.png")
        plt.close()
    
    # Prepare final effectiveness results
    effectiveness_results = {
        'baseline': {
            'yield': baseline_yield,
            'waste': baseline_waste,
            'effective_yield': baseline_effective
        },
        'scenario_improvements': scenario_improvements,
        'economic_impact': economic_impact
    }
    
    return effectiveness_results

def create_unified_prediction_system(model, data, selected_features, optimal_values=None):
    """
    Create a user-friendly prediction system that allows users to input
    agricultural parameters and get yield and waste predictions.
    
    Args:
        model: Trained unified model
        data: The integrated and feature-engineered dataset
        selected_features: List of selected features
        
    Returns:
        A prediction function that can be called with input parameters
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import warnings
    
    print("\nCreating unified prediction system...")
    
    if model is None or data is None or selected_features is None:
        print("Model, data, or feature list not available. Cannot create prediction system.")
        return None
    
    # Find the most important features to include in the interface
    # Focus on controllable parameters + crop selection
    key_input_params = [
        'nitrogen_mean', 'phosphorus_mean', 'potassium_mean',  # Fertilizer
        'ph_mean',  # Soil pH
        'rainfall_mean',  # Water/irrigation
        'temperature_mean', 'humidity_mean',  # Environmental conditions
        'crop'  # Crop selection
    ]
    
    # Filter to those that exist in our data
    available_input_params = [param for param in key_input_params if param in data.columns]
    
    print(f"Using {len(available_input_params)} parameters for the prediction interface:")
    print(f"  {', '.join(available_input_params)}")
    
    # Get ranges for numerical parameters
    param_ranges = {}
    for param in available_input_params:
        if param != 'crop' and pd.api.types.is_numeric_dtype(data[param]):
            param_ranges[param] = {
                'min': data[param].min(),
                'max': data[param].max(),
                'mean': data[param].mean(),
                'median': data[param].median(),
                'p25': data[param].quantile(0.25),
                'p75': data[param].quantile(0.75)
            }
    
    # Get list of crops if available
    available_crops = []
    if 'crop' in available_input_params:
        available_crops = data['crop'].unique().tolist()
        print(f"Found {len(available_crops)} crop types")
    
    # Create a function to pre-process inputs and make predictions
    def prediction_system(input_params):
        """
        Make yield and waste predictions based on input parameters.
        
        Args:
            input_params: Dictionary of input parameters
                Example: {
                    'nitrogen_mean': 80,
                    'phosphorus_mean': 40,
                    'potassium_mean': 60,
                    'ph_mean': 6.5,
                    'rainfall_mean': 200,
                    'temperature_mean': 25,
                    'humidity_mean': 70,
                    'crop': 'Rice'  # Optional
                }
        
        Returns:
            Dictionary of predictions and recommendations
        """
        # Create a DataFrame for the input
        input_df = pd.DataFrame([input_params])
        
        # Handle required features that are missing from input
        for feature in selected_features:
            if feature not in input_df.columns:
                # Check if it's an engineered feature we can calculate
                if feature == 'npk_balance' and all(col in input_df.columns for col in ['nitrogen_mean', 'phosphorus_mean', 'potassium_mean']):
                    # Calculate NPK balance
                    npk_sum = input_df['nitrogen_mean'] + input_df['phosphorus_mean'] + input_df['potassium_mean']
                    n_ratio = input_df['nitrogen_mean'] / npk_sum
                    p_ratio = input_df['phosphorus_mean'] / npk_sum
                    k_ratio = input_df['potassium_mean'] / npk_sum
                    input_df['npk_balance'] = 1 - (abs(n_ratio - 1/3) + abs(p_ratio - 1/3) + abs(k_ratio - 1/3))
                    
                elif feature == 'soil_fertility' and all(col in input_df.columns for col in ['nitrogen_mean', 'phosphorus_mean', 'potassium_mean']):
                    # Calculate soil fertility
                    # We'll use the ranges from the training data for normalization
                    n_max = param_ranges['nitrogen_mean']['max']
                    p_max = param_ranges['phosphorus_mean']['max']
                    k_max = param_ranges['potassium_mean']['max']
                    
                    input_df['soil_fertility'] = (
                        input_df['nitrogen_mean'] / n_max +
                        input_df['phosphorus_mean'] / p_max +
                        input_df['potassium_mean'] / k_max
                    ) / 3
                    
                elif feature == 'temp_humidity_index' and all(col in input_df.columns for col in ['temperature_mean', 'humidity_mean']):
                    # Calculate temperature-humidity index
                    input_df['temp_humidity_index'] = input_df['temperature_mean'] * input_df['humidity_mean'] / 100
                    
                elif feature == 'aridity_index' and all(col in input_df.columns for col in ['rainfall_mean', 'temperature_mean']):
                    # Calculate aridity index
                    input_df['aridity_index'] = input_df['rainfall_mean'] / (input_df['temperature_mean'] + 1)
                    
                elif feature == 'ph_optimality' and 'ph_mean' in input_df.columns:
                    # Calculate pH optimality
                    input_df['ph_optimality'] = 1 - abs((input_df['ph_mean'] - 6.5) / 3)
                    input_df['ph_optimality'] = input_df['ph_optimality'].clip(0, 1)
                    
                elif feature.startswith('crop_cat_') and 'crop' in input_df.columns:
                    # This is a crop category feature, set to 0 by default
                    input_df[feature] = 0
                    
                elif feature.startswith('is_') and 'season' in input_df.columns:
                    # This is a season indicator, set to 0 by default
                    input_df[feature] = 0
                    
                else:
                    # For other missing features, use median from training data if available
                    # Otherwise set to 0
                    if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
                        input_df[feature] = data[feature].median()
                    else:
                        input_df[feature] = 0
        
        # Select only the needed features
        feature_subset = [f for f in selected_features if f in input_df.columns]
        input_features = input_df[feature_subset]
        
        # Make predictions
        try:
            predictions = model.predict(input_features)
            
            production_pred = np.expm1(predictions[0, 0])  # Undo log1p
            area_val = input_features['area'].values[0] if 'area' in input_features.columns else 1.0
            predicted_yield = production_pred / area_val
            predicted_waste = predictions[0, 1]
            effective_yield = predicted_yield * (1 - predicted_waste/100)
            
            # Generate recommendations
            recommendations = []
            
            # Compare to optimal values from sensitivity analysis
            if optimal_values:
                for param, optimal in optimal_values.items():
                    if param in input_params:
                        current = input_params[param]
                        # Calculate how far from optimal
                        diff_pct = abs((current - optimal) / optimal) * 100
                        
                        if diff_pct > 15:  # If more than 15% off from optimal
                            if current < optimal:
                                recommendations.append(f"Consider increasing {param} from {current:.2f} to near {optimal:.2f} for better results.")
                            else:
                                recommendations.append(f"Consider decreasing {param} from {current:.2f} to near {optimal:.2f} for better results.")
            
            # Generate crop-specific recommendations
            if 'crop' in input_params and input_params['crop']:
                crop = input_params['crop']
                
                # Provide NPK recommendations based on crop needs
                if 'nitrogen_mean' in input_params and 'phosphorus_mean' in input_params and 'potassium_mean' in input_params:
                    n = input_params['nitrogen_mean']
                    p = input_params['phosphorus_mean']
                    k = input_params['potassium_mean']
                    
                    # Get ideal NPK for this crop (if available)
                    crop_subset = data[data['crop'] == crop]
                    if len(crop_subset) > 10:  # Only if we have enough data
                        ideal_n = crop_subset['nitrogen_mean'].median()
                        ideal_p = crop_subset['phosphorus_mean'].median()
                        ideal_k = crop_subset['potassium_mean'].median()
                        
                        # Compare and make recommendations
                        if abs(n - ideal_n) / ideal_n > 0.2:  # If more than 20% off
                            if n < ideal_n:
                                recommendations.append(f"{crop} typically needs more nitrogen. Consider increasing from {n:.1f} to near {ideal_n:.1f}.")
                            else:
                                recommendations.append(f"{crop} typically needs less nitrogen. Consider decreasing from {n:.1f} to near {ideal_n:.1f}.")
                                
                        if abs(p - ideal_p) / ideal_p > 0.2:
                            if p < ideal_p:
                                recommendations.append(f"{crop} typically needs more phosphorus. Consider increasing from {p:.1f} to near {ideal_p:.1f}.")
                            else:
                                recommendations.append(f"{crop} typically needs less phosphorus. Consider decreasing from {p:.1f} to near {ideal_p:.1f}.")
                                
                        if abs(k - ideal_k) / ideal_k > 0.2:
                            if k < ideal_k:
                                recommendations.append(f"{crop} typically needs more potassium. Consider increasing from {k:.1f} to near {ideal_k:.1f}.")
                            else:
                                recommendations.append(f"{crop} typically needs less potassium. Consider decreasing from {k:.1f} to near {ideal_k:.1f}.")
            
            # If no crop was specified, recommend one
            elif 'crop' not in input_params or not input_params['crop']:
                # Create test cases for all available crops
                if available_crops:
                    best_crop = None
                    best_effective_yield = 0
                    
                    for crop in available_crops:
                        # Create a copy of the input
                        crop_input = input_params.copy()
                        crop_input['crop'] = crop
                        
                        # Get prediction for this crop
                        crop_result = prediction_system(crop_input)
                        
                        # Check if this crop gives better effective yield
                        if crop_result['predictions']['effective_yield'] > best_effective_yield:
                            best_effective_yield = crop_result['predictions']['effective_yield']
                            best_crop = crop
                    
                    if best_crop:
                        recommendations.append(f"For your conditions, {best_crop} might be the optimal crop choice with an estimated effective yield of {best_effective_yield:.2f}.")
            
            # Return predictions and recommendations
            result = {
                'predictions': {
                    'yield': predicted_yield,
                    'food_loss_pct': predicted_waste,
                    'effective_yield': effective_yield
                },
                'recommendations': recommendations
            }
            
            # Include the crop if specified
            if 'crop' in input_params and input_params['crop']:
                result['crop'] = input_params['crop']
            
            return result
            
        except Exception as e:
            warnings.warn(f"Prediction error: {e}")
            return {
                'error': f"Prediction failed: {e}",
                'predictions': {
                    'yield': None,
                    'food_loss_pct': None,
                    'effective_yield': None
                }
            }
    
    # Create helper function to get parameter ranges (for UI)
    def get_param_ranges():
        """Get the valid ranges for all parameters"""
        return {
            'parameter_ranges': param_ranges,
            'available_crops': available_crops
        }
    
    # Attach the helper function to the main prediction function
    prediction_system.get_param_ranges = get_param_ranges
    
    # Print example usage
    print("\nPrediction system created successfully!")
    print("Example usage:")
    print("  result = prediction_system({")
    for param in available_input_params:
        if param != 'crop':
            print(f"      '{param}': {param_ranges[param]['median']:.2f},")
        else:
            if available_crops:
                print(f"      '{param}': '{available_crops[0]}',")
    print("  })")
    
    return prediction_system

def main(n_trials=50, n_iterations=100, fast_mode=False):
    """
    Main function to run the entire unified analysis pipeline.
    
    Args:
        n_trials: Number of hyperparameter optimization trials
        n_iterations: Number of agricultural parameter optimization iterations
        fast_mode: Whether to run a quick version for testing
        
    Returns:
        Dictionary of results including model, optimized parameters, and prediction system
    """
    print("\n" + "=" * 80)
    print("STARTING UNIFIED AGRICULTURAL OPTIMIZATION PIPELINE")
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

    print("\nAvailable columns in crop_yield_data:")
    print(crop_yield_data.columns)

    try:
        output_path = Path("crop_yield_data_preview.csv")
        crop_yield_data.to_csv(output_path, index=False)
        output_path
    except Exception as e:
        str(e)
    
    # Step 5: Feature engineering and selection
    print("\n" + "=" * 50)
    print("STEP 5: FEATURE ENGINEERING AND SELECTION")
    print("=" * 50)
    engineered_data, selected_features = engineer_and_select_features(integrated_data)

    try:
        output_path = Path("engineered_data_preview.csv")
        engineered_data.to_csv(output_path, index=False)
        output_path
    except Exception as e:
        str(e)
    
    print(selected_features)
    
    # Step 6: Develop unified model
    print("\n" + "=" * 50)
    print("STEP 6: UNIFIED MODEL DEVELOPMENT WITH HYPERPARAMETER OPTIMIZATION")
    print("=" * 50)
    unified_model, model_results, test_sets = develop_unified_model(
        engineered_data, selected_features, n_trials=n_trials
    )
    
    # Step 7: Evaluate unified model
    print("\n" + "=" * 50)
    print("STEP 7: UNIFIED MODEL EVALUATION")
    print("=" * 50)
    evaluation_metrics = evaluate_unified_model(unified_model, test_sets, engineered_data)
    
    # Step 8: Optimize agricultural parameters with unified model
    print("\n" + "=" * 50)
    print("STEP 8: AGRICULTURAL PARAMETER OPTIMIZATION")
    print("=" * 50)
    optimization_results = optimize_agricultural_parameters(
        unified_model, test_sets, engineered_data, selected_features, n_iterations=n_iterations
    )

    print(selected_features)
    
    # Step 9: Perform sensitivity analysis
    print("\n" + "=" * 50)
    print("STEP 9: UNIFIED SENSITIVITY ANALYSIS")
    print("=" * 50)
    sensitivity_results, optimal_values = perform_unified_sensitivity_analysis(
        unified_model, test_sets, engineered_data
    )
    
    # Step 10: Analyze model effectiveness
    print("\n" + "=" * 50)
    print("STEP 10: MODEL EFFECTIVENESS ANALYSIS")
    print("=" * 50)
    effectiveness_results = analyze_model_effectiveness(
        unified_model, test_sets, optimization_results
    )
    
    # Step 11: Create unified prediction system
    print("\n" + "=" * 50)
    print("STEP 11: UNIFIED PREDICTION SYSTEM")
    print("=" * 50)
    prediction_system = create_unified_prediction_system(
        unified_model, engineered_data, selected_features, optimal_values
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print(f"UNIFIED PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_time:.2f} SECONDS")
    print("=" * 80)
    
    # Generate summary report
    print("\nGenerating summary report...")
    
    # Create summary directory
    summary_dir = "summary"
    os.makedirs(summary_dir, exist_ok=True)
    
    # Model performance summary
    with open(f"{summary_dir}/unified_model_summary.txt", "w") as f:
        f.write("UNIFIED MODEL PERFORMANCE SUMMARY\n")
        f.write("===============================\n\n")
        
        f.write(f"Algorithm: {model_results['algorithm']}\n\n")
        
        f.write("Yield Prediction Metrics:\n")
        for metric, value in model_results['metrics']['yield'].items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\nWaste Prediction Metrics:\n")
        for metric, value in model_results['metrics']['waste'].items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\nEffective Yield Prediction Metrics:\n")
        for metric, value in model_results['metrics']['effective_yield'].items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    # Optimization summary
    if optimization_results:
        with open(f"{summary_dir}/optimization_summary.txt", "w") as f:
            f.write("AGRICULTURAL PARAMETER OPTIMIZATION SUMMARY\n")
            f.write("==========================================\n\n")
            
            for scenario_name, result in optimization_results.items():
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
    
    # Return the final results
    return {
        'integrated_data': engineered_data,
        'selected_features': selected_features,
        'unified_model': unified_model,
        'model_results': model_results,
        'optimization_results': optimization_results,
        'sensitivity_results': sensitivity_results,
        'optimal_values': optimal_values,
        'effectiveness_results': effectiveness_results,
        'prediction_system': prediction_system
    }


if __name__ == "__main__":
    # Run the pipeline with fast mode for development
    results = main(fast_mode=True)
    
    # For production, uncomment the line below:
    # results = main(n_trials=100, n_iterations=200)
    
    print("\nUnified model training and evaluation completed successfully!")
    print("Models saved to 'models' directory")
    print("Visualizations saved to 'visualizations' directory")
    print("Summary reports saved to 'summary' directory")
    
    # Run a quick test of the unified prediction system
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
        print(f"  Recommended crop: {prediction['crop']}")
        for key, value in prediction['predictions'].items():
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}" + ("%" if key == 'food_loss_pct' else ""))
        
        print("\nUse the trained unified model as follows:")
        print("  1. Import the module")
        print("  2. Call the main() function to train the unified model")
        print("  3. Access the prediction_system from the returned results")
        print("  4. Pass your input parameters to the prediction system")
    except Exception as e:
        print(f"\nError during sample prediction: {e}")
        print("Model training and evaluation still completed successfully.")

