"""
Multi-Week Sales Forecaster - Fixed Recursive Logic
====================================================

Three distinct forecasting strategies:

1. direct_multi: Four independent models (no recursion)
2. recursive_single: One model trained/reused with simulated lag features
3. recursive_multi: Four chained models with proper lag simulation

Key fix: Recursive methods now properly simulate prediction-time features
during training to avoid train-test mismatch.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MultiWeekForecaster:
    """
    Production-ready multi-week sales forecasting with 3 distinct strategies.
    """
    
    def __init__(
        self, 
        horizons: List[int] = [1, 2, 3, 4],
        strategy: str = 'direct_multi',
        model_params: Optional[Dict] = None,
        random_seed: int = 42
    ):
        """
        Args:
            horizons: Weeks ahead to predict [1, 2, 3, 4]
            strategy: 'direct_multi', 'recursive_single', or 'recursive_multi'
            model_params: XGBoost hyperparameters
            random_seed: For reproducibility
        """
        valid_strategies = ['direct_multi', 'recursive_single', 'recursive_multi']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}, got '{strategy}'")
        
        self.horizons = horizons
        self.strategy = strategy
        self.random_seed = random_seed
        
        # Default XGBoost params
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_seed,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # State
        self.models: Dict[int, XGBRegressor] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.target_col: str = 'weekly_sales'
        
    def _identify_features(self, df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[List[str], List[str]]:
        """Identify feature columns and categorical features."""
        default_exclude = [
            self.target_col, 'week_start', 'week_end', 'ean', 'machine_key', 
            'product_name', 'provider', 'refiller', 'customer_id', 'date_key'
        ]
        # Also exclude any target columns
        for h in self.horizons:
            default_exclude.append(f'target_week_{h}')
        
        exclude_cols = list(set(default_exclude + exclude_cols))
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        categorical = [
            col for col in feature_cols 
            if df[col].dtype == 'object' or df[col].dtype.name == 'category'
        ]
        
        return feature_cols, categorical
    
    def _encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        df = df.copy()
        
        for col in self.categorical_features:
            if fit:
                le = LabelEncoder()
                df[col] = df[col].astype(str).fillna('_missing_')
                le.fit(df[col])
                self.label_encoders[col] = le
                df[col] = le.transform(df[col])
            else:
                le = self.label_encoders[col]
                df[col] = df[col].astype(str).fillna('_missing_')
                unseen_mask = ~df[col].isin(le.classes_)
                if unseen_mask.any():
                    df.loc[unseen_mask, col] = le.classes_[0]
                df[col] = le.transform(df[col])
        
        return df
    
    def _create_multi_week_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target columns for each horizon."""
        df = df.sort_values(['machine_key', 'ean', 'week_start']).copy()
        
        for horizon in self.horizons:
            target_col = f'target_week_{horizon}'
            df[target_col] = df.groupby(['machine_key', 'ean'])[self.target_col].shift(-horizon)
        
        return df
    
    def _prepare_data(
        self, 
        df: pd.DataFrame, 
        exclude_cols: List[str] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and targets."""
        exclude_cols = exclude_cols or []
        
        if fit:
            self.feature_names, self.categorical_features = self._identify_features(df, exclude_cols)
            print(f"✓ Identified {len(self.feature_names)} features")
            print(f"  - Categorical: {len(self.categorical_features)}")
            print(f"  - Numerical: {len(self.feature_names) - len(self.categorical_features)}")
        
        df = self._create_multi_week_targets(df)
        df = self._encode_features(df, fit=fit)
        
        X = df[self.feature_names].copy()
        y_cols = [f'target_week_{h}' for h in self.horizons]
        y = df[y_cols].copy()
        
        ######## CHANGES THIS FROM MOAAZ PROD ########
        # Only keep rows where ALL targets are available (ONLY during training)
        if fit:
            valid_mask = y.notna().all(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Filter to valid dtypes
        allowed_types = ['number', 'bool', 'category']
        valid_cols = X.select_dtypes(include=allowed_types).columns
        invalid_cols = [c for c in X.columns if c not in valid_cols]

        if invalid_cols:
            print(f"⚠ Ignored {len(invalid_cols)} columns with unsupported dtypes")
            X = X[valid_cols]
            
        # Update feature names to match valid columns
        if fit:
            self.feature_names = list(valid_cols)

        ###### ADDED THIS IF STATEMENT TO PRINT THE CORRECT MESSAGE FOR TRAINING AND PREDICTION
        if fit:
            print(f"✓ Prepared {len(X):,} samples ({valid_mask.sum() / len(df):.1%} of input)")
        else:
            print(f"✓ Prepared {len(X):,} samples for prediction")
            
        return X, y

    def fit(self, train_df: pd.DataFrame, exclude_cols: List[str] = None) -> 'MultiWeekForecaster':
        """Train forecasting models."""
        print(f"\n{'='*70}")
        print(f"Training: {self.strategy.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        
        X_train, y_train = self._prepare_data(train_df, exclude_cols, fit=True)
        
        if self.strategy == 'direct_multi':
            self._fit_direct_multi(X_train, y_train)
        elif self.strategy == 'recursive_single':
            self._fit_recursive_single(X_train, y_train)
        elif self.strategy == 'recursive_multi':
            self._fit_recursive_multi(X_train, y_train)
        
        print(f"✓ Training complete\n")
        return self
    
    # =========================================================================
    # DIRECT MULTI: Four independent models
    # =========================================================================
    def _fit_direct_multi(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Strategy: Four separate XGBoost models
        Input: Features from week t-1 (same for all models)
        Output: Each model predicts ONE specific week
        """
        print(f"\nTraining {len(self.horizons)} independent models...")
        print(f"  Input: {X.shape[1]} features from week t-1 (same for all)")
        
        for horizon in self.horizons:
            target_col = f'target_week_{horizon}'
            
            model = XGBRegressor(**self.model_params)
            model.fit(X, y[target_col])
            self.models[horizon] = model
            
            train_pred = model.predict(X)
            train_mae = mean_absolute_error(y[target_col], train_pred)
            print(f"  Model {horizon} (week +{horizon}): Train MAE = {train_mae:.3f}")
    
    # =========================================================================
    # RECURSIVE SINGLE: One model reused with lag feature
    # =========================================================================
    def _fit_recursive_single(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Strategy: One model trained with 'prev_sales' feature, reused iteratively.
        
        Training approach:
        - Create 'prev_sales' feature using out-of-fold predictions to simulate
          what will happen during recursive prediction
        - For first fold, use 0 (no prior prediction available)
        - Train single model on week+1 with this simulated lag feature
        
        Prediction:
        - Week +1: prev_sales = 0 (no prior prediction)
        - Week +2: prev_sales = pred_week_1
        - Week +3: prev_sales = pred_week_2
        - Week +4: prev_sales = pred_week_3
        """
        print(f"\nTraining single recursive model...")
        print(f"  Input: {X.shape[1]} base features + 'prev_sales' (simulated)")
        print(f"  Using 3-fold CV to create unbiased lag feature")
        
        target_col = 'target_week_1'
        
        # Create simulated 'prev_sales' using cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_seed)
        prev_sales = np.zeros(len(X))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Train temporary model on fold
            temp_model = XGBRegressor(**self.model_params)
            temp_model.fit(X.iloc[train_idx], y[target_col].iloc[train_idx])
            
            # Generate predictions for validation fold
            prev_sales[val_idx] = temp_model.predict(X.iloc[val_idx])
        
        # Now train final model with simulated prev_sales feature
        X_with_lag = X.copy()
        X_with_lag['prev_sales'] = prev_sales
        
        model = XGBRegressor(**self.model_params)
        model.fit(X_with_lag, y[target_col])
        
        # Store for all horizons (will reuse recursively)
        for horizon in self.horizons:
            self.models[horizon] = model
        
        train_pred = model.predict(X_with_lag)
        train_mae = mean_absolute_error(y[target_col], train_pred)
        print(f"  Train MAE (week +1 with simulated lag): {train_mae:.3f}")
        print(f"  ⚠ Errors will compound for weeks +2, +3, +4")
    
    # =========================================================================
    # RECURSIVE MULTI: Four chained models with proper simulation
    # =========================================================================
    def _fit_recursive_multi(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Strategy: Four separate models, each uses previous model's predictions.
        
        Model 1: base features → predicts week+1 (no lag)
        Model 2: base features + Model1_pred → predicts week+2
        Model 3: base features + Model2_pred → predicts week+3
        Model 4: base features + Model3_pred → predicts week+4
        
        Training: Use out-of-fold predictions from previous models to avoid leakage.
        """
        print(f"\nTraining {len(self.horizons)} chained models...")
        print(f"  Base input: {X.shape[1]} features")
        print(f"  Using OOF predictions to simulate recursive features")
        
        # Dictionary to store OOF predictions from each model
        oof_predictions = {}
        
        for i, horizon in enumerate(self.horizons):
            target_col = f'target_week_{horizon}'
            
            # Step 1: Generate OOF predictions for this model
            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_seed)
            oof_pred = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X):
                # Build training features
                X_train_fold = X.iloc[train_idx].copy()
                X_val_fold = X.iloc[val_idx].copy()
                
                # Add lag feature if not Model 1
                if horizon == 1:
                    # Model 1: no lag feature
                    pass
                elif horizon == 2:
                    # Model 2: use Model 1's OOF predictions as lag
                    X_train_fold['prev_sales'] = oof_predictions[1][train_idx]
                    X_val_fold['prev_sales'] = oof_predictions[1][val_idx]
                else:
                    # Models 3+: use previous model's OOF as lag
                    prev_h = horizon - 1
                    X_train_fold['prev_sales'] = oof_predictions[prev_h][train_idx]
                    X_val_fold['prev_sales'] = oof_predictions[prev_h][val_idx]
                
                # Train temporary model on fold
                temp_model = XGBRegressor(**self.model_params)
                temp_model.fit(X_train_fold, y[target_col].iloc[train_idx])
                
                # Generate OOF prediction for this fold
                oof_pred[val_idx] = temp_model.predict(X_val_fold)
            
            # Store OOF predictions for next model
            oof_predictions[horizon] = oof_pred
            
            # Step 2: Train final model with all data
            X_train = X.copy()
            if horizon > 1:
                # Add lag feature using OOF predictions
                prev_h = horizon - 1
                X_train['prev_sales'] = oof_predictions[prev_h]
                print(f"\n  Model {i+1} (week +{horizon}): {X_train.shape[1]} features (base + prev_sales)")
            else:
                print(f"\n  Model 1 (week +1): {X_train.shape[1]} features")
            
            # Train final model
            model = XGBRegressor(**self.model_params)
            model.fit(X_train, y[target_col])
            self.models[horizon] = model
            
            # Train prediction for metrics
            train_pred = model.predict(X_train)
            train_mae = mean_absolute_error(y[target_col], train_pred)
            print(f"    Train MAE: {train_mae:.3f}")
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Generate multi-week predictions."""
        print(f"\n{'='*70}")
        print(f"Generating Predictions: {self.strategy.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        
        X_test, y_test = self._prepare_data(test_df, fit=False)
        
        if self.strategy == 'direct_multi':
            predictions = self._predict_direct_multi(X_test)
        elif self.strategy == 'recursive_single':
            predictions = self._predict_recursive_single(X_test)
        elif self.strategy == 'recursive_multi':
            predictions = self._predict_recursive_multi(X_test)
        
        # Combine with actuals
        results = pd.DataFrame(predictions)
        for horizon in self.horizons:
            results[f'actual_week_{horizon}'] = y_test[f'target_week_{horizon}'].values
        
        print(f"✓ Generated {len(results):,} predictions\n")
        return results
    
    def _predict_direct_multi(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Independent predictions from 4 models."""
        predictions = {}
        for horizon in self.horizons:
            model = self.models[horizon]
            predictions[f'pred_week_{horizon}'] = model.predict(X)
        return predictions
    
    def _predict_recursive_single(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Recursive predictions with same model.
        Each prediction uses previous prediction as 'prev_sales' feature.
        """
        predictions = {}
        model = self.models[1]
        
        # Week +1: prev_sales = 0 (no prior prediction)
        X_with_lag = X.copy()
        X_with_lag['prev_sales'] = 0
        pred_1 = model.predict(X_with_lag)
        predictions['pred_week_1'] = pred_1
        
        # Weeks +2, +3, +4: Use previous prediction as prev_sales
        for horizon in self.horizons[1:]:
            X_with_lag = X.copy()
            X_with_lag['prev_sales'] = predictions[f'pred_week_{horizon-1}']
            pred = model.predict(X_with_lag)
            predictions[f'pred_week_{horizon}'] = pred
        
        return predictions
    
    def _predict_recursive_multi(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Recursive predictions with chained models.
        Each model uses previous model's prediction as 'prev_sales'.
        """
        predictions = {}
        
        for i, horizon in enumerate(self.horizons):
            model = self.models[horizon]
            
            if horizon == 1:
                # Model 1: just base features
                X_current = X.copy()
            else:
                # Models 2-4: add previous prediction as prev_sales
                X_current = X.copy()
                X_current['prev_sales'] = predictions[f'pred_week_{horizon-1}']
            
            pred = model.predict(X_current)
            predictions[f'pred_week_{horizon}'] = pred
        
        return predictions
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    def evaluate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive evaluation across all horizons."""
        print(f"\n{'='*70}")
        print(f"Evaluation Metrics")
        print(f"{'='*70}\n")
        
        metrics = []
        
        for horizon in self.horizons:
            pred_col = f'pred_week_{horizon}'
            actual_col = f'actual_week_{horizon}'
            
            y_true = predictions[actual_col].values
            y_pred = predictions[pred_col].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            metrics.append({
                'horizon': f'Week +{horizon}',
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'samples': len(y_true)
            })
            
            print(f"Week +{horizon}:")
            print(f"  MAE:  {mae:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  R²:   {r2:.3f}")
            print(f"  MAPE: {mape:.1f}%")
            print()
        
        avg_mae = np.mean([m['mae'] for m in metrics])
        avg_r2 = np.mean([m['r2'] for m in metrics])
        
        print(f"{'─'*70}")
        print(f"4-Week Average: MAE {avg_mae:.3f}, R² {avg_r2:.3f}")
        print(f"{'─'*70}\n")
        
        return pd.DataFrame(metrics)
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    def save(self, path: Path):
        """Save models and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for key, model in self.models.items():
            model_path = path / f'model_{key}.pkl'
            joblib.dump(model, model_path)
        
        metadata = {
            'horizons': self.horizons,
            'strategy': self.strategy,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'label_encoders': self.label_encoders,
            'target_col': self.target_col
        }
        joblib.dump(metadata, path / 'metadata.pkl')
        
        print(f"✓ Models saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'MultiWeekForecaster':
        """Load models and metadata."""
        path = Path(path)
        
        metadata = joblib.load(path / 'metadata.pkl')
        
        forecaster = cls(
            horizons=metadata['horizons'],
            strategy=metadata['strategy'],
            model_params=metadata['model_params']
        )
        
        forecaster.feature_names = metadata['feature_names']
        forecaster.categorical_features = metadata['categorical_features']
        forecaster.label_encoders = metadata['label_encoders']
        forecaster.target_col = metadata['target_col']
        
        # Load models
        for horizon in forecaster.horizons:
            model_path = path / f'model_{horizon}.pkl'
            forecaster.models[horizon] = joblib.load(model_path)
        
        print(f"✓ Models loaded from {path}")
        return forecaster