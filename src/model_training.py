"""
Model Training Pipeline for Predictive Maintenance
Trains classification, regression, and anomaly detection models
"""
import pandas as pd
import numpy as np
import logging
import os
import joblib
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from xgboost import XGBClassifier, XGBRegressor

# Handle imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from config_file import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate predictive maintenance models"""
    
    def __init__(self, features_path=None):
        """Initialize trainer with feature data"""
        if features_path is None:
            features_path = os.path.join(
                Config.DATA_DIR, 'processed', 'features_with_labels.csv'
            )
        
        self.features_path = features_path
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Create models directory if not exists
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    def load_data(self):
        """Load processed features and labels"""
        logger.info(f"Loading data from {self.features_path}")
        
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        df = pd.read_csv(self.features_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Show label distribution
        logger.info(f"\nFailure label distribution:")
        logger.info(df['will_fail_7d'].value_counts())
        logger.info(f"Failure rate: {df['will_fail_7d'].mean()*100:.2f}%")
        
        if 'rul' in df.columns:
            rul_count = df['rul'].notna().sum()
            logger.info(f"\nRUL samples: {rul_count}")
            logger.info(f"RUL range: {df['rul'].min():.1f} to {df['rul'].max():.1f} days")
        
        return df
    
    def prepare_classification_data(self, df):
        """Prepare data for failure classification"""
        logger.info("\nPreparing classification data...")
        
        # Separate features and target
        X = df.drop(['machine_id', 'will_fail_7d', 'rul'], axis=1, errors='ignore')
        y = df['will_fail_7d']
        
        # Store feature names
        self.feature_columns = list(X.columns)
        
        # Handle missing values and infinity
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Split data (stratified for imbalanced classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Replace any inf in train/test splits
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Replace any inf in scaled data
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
        
        # Handle class imbalance with SMOTE
        logger.info("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        logger.info(f"Training set: {X_train_balanced.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        logger.info(f"Class distribution after SMOTE: {np.bincount(y_train_balanced)}")
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test
    
    def prepare_regression_data(self, df):
        """Prepare data for RUL regression"""
        logger.info("\nPreparing regression data...")
        
        # Filter samples with RUL values
        df_rul = df[df['rul'].notna()].copy()
        logger.info(f"Using {len(df_rul)} samples with RUL values")
        
        X = df_rul.drop(['machine_id', 'will_fail_7d', 'rul'], axis=1, errors='ignore')
        y = df_rul['rul']
        
        # Handle missing values and infinity
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Replace any inf in train/test splits
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        
        # Scale features (use same scaler as classification if possible)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Replace any inf in scaled data
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
        
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_classifier(self, X_train, X_test, y_train, y_test):
        """Train failure classification model"""
        logger.info("\n" + "="*60)
        logger.info("Training Failure Classification Model")
        logger.info("="*60)
        
        # Random Forest Classifier
        logger.info("\n1. Random Forest Classifier")
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf_clf.fit(X_train, y_train)
        self._evaluate_classifier(rf_clf, X_test, y_test, "Random Forest")
        
        # XGBoost Classifier
        logger.info("\n2. XGBoost Classifier")
        xgb_clf = XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_clf.fit(X_train, y_train)
        self._evaluate_classifier(xgb_clf, X_test, y_test, "XGBoost")
        
        # Choose best model (let's use XGBoost as it typically performs better)
        best_model = xgb_clf
        self.models['classifier'] = best_model
        
        # Save model
        model_path = os.path.join(Config.MODEL_DIR, 'classifier.pkl')
        joblib.dump(best_model, model_path)
        logger.info(f"\n✅ Classifier saved to {model_path}")
        
        # Save feature importance
        self._plot_feature_importance(best_model, "Classification")
        
        return best_model
    
    def train_regressor(self, X_train, X_test, y_train, y_test):
        """Train RUL regression model"""
        logger.info("\n" + "="*60)
        logger.info("Training RUL Regression Model")
        logger.info("="*60)
        
        # Random Forest Regressor
        logger.info("\n1. Random Forest Regressor")
        rf_reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        rf_reg.fit(X_train, y_train)
        self._evaluate_regressor(rf_reg, X_test, y_test, "Random Forest")
        
        # XGBoost Regressor
        logger.info("\n2. XGBoost Regressor")
        xgb_reg = XGBRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_reg.fit(X_train, y_train)
        self._evaluate_regressor(xgb_reg, X_test, y_test, "XGBoost")
        
        # Choose best model
        best_model = xgb_reg
        self.models['regressor'] = best_model
        
        # Save model
        model_path = os.path.join(Config.MODEL_DIR, 'regressor.pkl')
        joblib.dump(best_model, model_path)
        logger.info(f"\n✅ Regressor saved to {model_path}")
        
        # Save feature importance
        self._plot_feature_importance(best_model, "Regression")
        
        return best_model
    
    def train_anomaly_detector(self, X_train):
        """Train anomaly detection model"""
        logger.info("\n" + "="*60)
        logger.info("Training Anomaly Detection Model")
        logger.info("="*60)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        
        iso_forest.fit(X_train)
        
        # Test on training data
        train_scores = iso_forest.score_samples(X_train)
        train_predictions = iso_forest.predict(X_train)
        
        anomaly_count = (train_predictions == -1).sum()
        logger.info(f"Detected {anomaly_count} anomalies in training data ({anomaly_count/len(X_train)*100:.2f}%)")
        
        self.models['anomaly_detector'] = iso_forest
        
        # Save model
        model_path = os.path.join(Config.MODEL_DIR, 'anomaly_detector.pkl')
        joblib.dump(iso_forest, model_path)
        logger.info(f"\n✅ Anomaly detector saved to {model_path}")
        
        return iso_forest
    
    def _evaluate_classifier(self, model, X_test, y_test, model_name):
        """Evaluate classification model"""
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        logger.info(f"\n{model_name} Performance:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # AUC-ROC
        auc_score = roc_auc_score(y_test, y_proba)
        logger.info(f"AUC-ROC Score: {auc_score:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
    
    def _evaluate_regressor(self, model, X_test, y_test, model_name):
        """Evaluate regression model"""
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"MAE:  {mae:.2f} days")
        logger.info(f"RMSE: {rmse:.2f} days")
        logger.info(f"R² Score: {r2:.4f}")
        
        # Additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        logger.info(f"MAPE: {mape:.2f}%")
    
    def _plot_feature_importance(self, model, model_type):
        """Plot and save feature importance"""
        if not hasattr(model, 'feature_importances_'):
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances - {model_type}')
        plt.barh(range(20), importances[indices])
        plt.yticks(range(20), [self.feature_columns[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        
        plot_path = os.path.join(Config.BASE_DIR, 'reports', 
                                f'feature_importance_{model_type.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {plot_path}")
        plt.close()
    
    def save_scaler(self):
        """Save the fitted scaler"""
        scaler_path = os.path.join(Config.MODEL_DIR, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✅ Scaler saved to {scaler_path}")
    
    def save_feature_columns(self):
        """Save feature column names"""
        feature_path = os.path.join(Config.MODEL_DIR, 'feature_columns.txt')
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.feature_columns))
        logger.info(f"✅ Feature columns saved to {feature_path}")
    
    def run_training_pipeline(self):
        """Execute complete training pipeline"""
        logger.info("\n" + "="*70)
        logger.info("PREDICTIVE MAINTENANCE MODEL TRAINING PIPELINE")
        logger.info("="*70)
        
        # Load data
        df = self.load_data()
        
        # Train classification model
        X_train_c, X_test_c, y_train_c, y_test_c = self.prepare_classification_data(df)
        self.train_classifier(X_train_c, X_test_c, y_train_c, y_test_c)
        
        # Train regression model
        X_train_r, X_test_r, y_train_r, y_test_r = self.prepare_regression_data(df)
        self.train_regressor(X_train_r, X_test_r, y_train_r, y_test_r)
        
        # Train anomaly detector
        self.train_anomaly_detector(X_train_c)
        
        # Save scaler and feature columns
        self.save_scaler()
        self.save_feature_columns()
        
        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info("\nModels saved in:", Config.MODEL_DIR)
        logger.info("  - classifier.pkl")
        logger.info("  - regressor.pkl")
        logger.info("  - anomaly_detector.pkl")
        logger.info("  - scaler.joblib")
        logger.info("  - feature_columns.txt")

def main():
    """Main execution function"""
    trainer = ModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()
