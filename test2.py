# new version
"""
Non-Technical Losses (NTL) Detection System for Power Distribution
Includes: Theft Detection, Simulation, ML/Statistical Methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PART 1: DATA SIMULATION - NTL and Normal Consumption Patterns
# ============================================================================

class PowerConsumptionSimulator:
    """Simulates power consumption data with NTL patterns"""
    
    def __init__(self, n_customers=1000, n_days=365):
        self.n_customers = n_customers
        self.n_days = n_days
        self.hours_per_day = 24
        
    def generate_normal_consumption(self, customer_type='residential'):
        """Generate normal consumption patterns"""
        n_records = self.n_customers * self.n_days
        
        # Base consumption by customer type
        base_consumption = {
            'residential': (5, 15),  # kWh per day (mean, std)
            'commercial': (50, 20),
            'industrial': (200, 50)
        }
        
        mean_consumption, std_consumption = base_consumption.get(customer_type, (10, 5))
        
        # Generate daily consumption with seasonal and weekly patterns
        data = []
        start_date = datetime(2023, 1, 1)
        
        for customer_id in range(self.n_customers):
            # Customer-specific baseline
            customer_baseline = np.random.normal(mean_consumption, std_consumption * 0.3)
            customer_baseline = max(1, customer_baseline)  # Ensure positive
            
            for day in range(self.n_days):
                current_date = start_date + timedelta(days=day)
                
                # Seasonal component (annual cycle)
                seasonal = 1 + 0.3 * np.sin(2 * np.pi * day / 365)
                
                # Weekly pattern (lower on weekends for commercial)
                if customer_type == 'commercial':
                    weekly = 0.7 if current_date.weekday() >= 5 else 1.0
                else:
                    weekly = 1.0
                
                # Daily noise
                noise = np.random.normal(1, 0.1)
                
                # Calculate consumption
                daily_consumption = customer_baseline * seasonal * weekly * noise
                
                # Hourly distribution (simplified)
                hourly_pattern = self._generate_hourly_pattern(customer_type)
                
                data.append({
                    'customer_id': f'C{customer_id:04d}',
                    'date': current_date,
                    'day_of_week': current_date.weekday(),
                    'month': current_date.month,
                    'daily_consumption_kwh': daily_consumption,
                    'peak_hour_consumption': daily_consumption * hourly_pattern,
                    'customer_type': customer_type,
                    'is_ntl': 0
                })
        
        return pd.DataFrame(data)
    
    def _generate_hourly_pattern(self, customer_type):
        """Generate realistic hourly consumption pattern"""
        if customer_type == 'residential':
            # Peak in morning and evening
            return np.random.choice([0.3, 0.4, 0.6, 0.8, 1.0], p=[0.2, 0.3, 0.2, 0.2, 0.1])
        elif customer_type == 'commercial':
            # Peak during business hours
            return np.random.choice([0.2, 0.5, 0.9, 1.0], p=[0.1, 0.2, 0.4, 0.3])
        else:
            # Industrial - more constant
            return np.random.choice([0.8, 0.9, 1.0], p=[0.3, 0.4, 0.3])
    
    def inject_ntl_patterns(self, df, ntl_ratio=0.15):
        """Inject various NTL patterns into the dataset"""
        n_ntl_customers = int(len(df['customer_id'].unique()) * ntl_ratio)
        ntl_customers = np.random.choice(
            df['customer_id'].unique(), 
            size=n_ntl_customers, 
            replace=False
        )
        
        df_ntl = df.copy()
        
        for customer in ntl_customers:
            mask = df_ntl['customer_id'] == customer
            ntl_type = np.random.choice(['theft', 'meter_tampering', 'irregular'])
            
            if ntl_type == 'theft':
                # Sudden drop in consumption (direct theft)
                start_day = np.random.randint(30, self.n_days - 30)
                theft_mask = mask & (df_ntl.index >= df_ntl[mask].index[start_day])
                reduction_factor = np.random.uniform(0.3, 0.7)
                df_ntl.loc[theft_mask, 'daily_consumption_kwh'] *= reduction_factor
                df_ntl.loc[theft_mask, 'peak_hour_consumption'] *= reduction_factor
                
            elif ntl_type == 'meter_tampering':
                # Gradual reduction with irregular patterns
                reduction = np.random.uniform(0.4, 0.8)
                irregular_pattern = np.random.normal(reduction, 0.1, mask.sum())
                irregular_pattern = np.clip(irregular_pattern, 0.2, 1.0)
                df_ntl.loc[mask, 'daily_consumption_kwh'] *= irregular_pattern
                
            else:  # irregular
                # Highly variable consumption (bypassing or faulty meter)
                df_ntl.loc[mask, 'daily_consumption_kwh'] *= np.random.uniform(0.1, 1.5, mask.sum())
            
            df_ntl.loc[mask, 'is_ntl'] = 1
        
        return df_ntl


# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

class NTLFeatureEngineer:
    """Extract features relevant for NTL detection"""
    
    @staticmethod
    def engineer_features(df):
        """Create features for NTL detection"""
        df = df.sort_values(['customer_id', 'date'])
        
        # Time-based features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter
        
        # Consumption statistics per customer
        customer_stats = df.groupby('customer_id').agg({
            'daily_consumption_kwh': ['mean', 'std', 'min', 'max', 'median']
        }).reset_index()
        customer_stats.columns = ['customer_id', 'consumption_mean', 'consumption_std', 
                                   'consumption_min', 'consumption_max', 'consumption_median']
        
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Deviation from customer mean
        df['consumption_deviation'] = (df['daily_consumption_kwh'] - df['consumption_mean']) / (df['consumption_std'] + 1e-6)
        
        # Rolling statistics (30-day window)
        df['consumption_rolling_mean_30d'] = df.groupby('customer_id')['daily_consumption_kwh'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        df['consumption_rolling_std_30d'] = df.groupby('customer_id')['daily_consumption_kwh'].transform(
            lambda x: x.rolling(window=30, min_periods=1).std()
        )
        
        # Change point detection features
        df['consumption_diff'] = df.groupby('customer_id')['daily_consumption_kwh'].diff()
        df['consumption_pct_change'] = df.groupby('customer_id')['daily_consumption_kwh'].pct_change()
        
        # Sudden drop indicator (potential theft start)
        df['sudden_drop'] = ((df['consumption_pct_change'] < -0.3) & 
                             (df['consumption_pct_change'].notna())).astype(int)
        
        # Coefficient of variation
        df['cv'] = df['consumption_std'] / (df['consumption_mean'] + 1e-6)
        
        # Ratio to peak
        df['ratio_to_max'] = df['daily_consumption_kwh'] / (df['consumption_max'] + 1e-6)
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df


# ============================================================================
# PART 3: DETECTION MODELS
# ============================================================================

class NTLDetector:
    """Multiple detection algorithms for NTL"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_data(self, df, feature_cols):
        """Prepare data for modeling"""
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        y = df['is_ntl'].values if 'is_ntl' in df.columns else None
        
        return X, y
    
    def train_supervised_models(self, X_train, y_train):
        """Train supervised ML models"""
        print("Training Supervised Models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Random Forest
        print("  - Random Forest Classifier")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        
        # Gradient Boosting
        print("  - Gradient Boosting Classifier")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['gradient_boosting'] = gb_model
        
        return self
    
    def train_unsupervised_models(self, X_train):
        """Train unsupervised anomaly detection models"""
        print("Training Unsupervised Models...")
        
        X_train_scaled = self.scaler.transform(X_train)
        
        # Isolation Forest
        print("  - Isolation Forest")
        iso_forest = IsolationForest(
            contamination=0.15,
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_train_scaled)
        self.models['isolation_forest'] = iso_forest
        
        # One-Class SVM
        print("  - One-Class SVM")
        oc_svm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.15
        )
        oc_svm.fit(X_train_scaled)
        self.models['one_class_svm'] = oc_svm
        
        return self
    
    def train_statistical_detector(self, X_train):
        """Statistical anomaly detection based on Z-scores"""
        print("Training Statistical Detector...")
        
        # Calculate statistics for each feature
        self.statistical_params = {
            'mean': np.mean(X_train, axis=0),
            'std': np.std(X_train, axis=0)
        }
        
        return self
    
    def predict(self, X_test, model_name):
        """Make predictions using specified model"""
        X_test_scaled = self.scaler.transform(X_test)
        
        if model_name in ['random_forest', 'gradient_boosting']:
            predictions = self.models[model_name].predict(X_test_scaled)
            probabilities = self.models[model_name].predict_proba(X_test_scaled)[:, 1]
            
        elif model_name in ['isolation_forest', 'one_class_svm']:
            # Convert -1/1 to 0/1
            predictions = self.models[model_name].predict(X_test_scaled)
            predictions = (predictions == -1).astype(int)
            probabilities = -self.models[model_name].score_samples(X_test_scaled)
            
        elif model_name == 'statistical':
            # Z-score based detection
            z_scores = np.abs(
                (X_test - self.statistical_params['mean']) / 
                (self.statistical_params['std'] + 1e-6)
            )
            # Anomaly if any feature has z-score > 3
            predictions = (np.max(z_scores, axis=1) > 3).astype(int)
            probabilities = np.max(z_scores, axis=1)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return predictions, probabilities
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance from tree-based models"""
        if model_name in ['random_forest', 'gradient_boosting']:
            importances = self.models[model_name].feature_importances_
            return pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            return None


# ============================================================================
# PART 4: EVALUATION AND VISUALIZATION
# ============================================================================

class NTLEvaluator:
    """Evaluate and visualize NTL detection results"""
    
    @staticmethod
    def evaluate_model(y_true, y_pred, y_prob, model_name):
        """Comprehensive model evaluation"""
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {model_name}")
        print(f"{'='*60}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Normal', 'NTL'],
                                   digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Normal  NTL")
        print(f"Actual Normal  {cm[0,0]:6d}  {cm[0,1]:4d}")
        print(f"       NTL     {cm[1,0]:6d}  {cm[1,1]:4d}")
        
        # Additional Metrics
        if len(np.unique(y_true)) > 1:
            auc_score = roc_auc_score(y_true, y_prob)
            f1 = f1_score(y_true, y_pred)
            print(f"\nROC-AUC Score: {auc_score:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Detection rate and false positive rate
            detection_rate = cm[1,1] / (cm[1,0] + cm[1,1])
            false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1])
            print(f"Detection Rate (Recall): {detection_rate:.4f}")
            print(f"False Positive Rate: {false_positive_rate:.4f}")
        
        return {
            'confusion_matrix': cm,
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
    
    @staticmethod
    def plot_roc_curves(results_dict, y_true):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, (y_pred, y_prob) in results_dict.items():
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_score = roc_auc_score(y_true, y_prob)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - NTL Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_consumption_patterns(df, customer_id):
        """Visualize consumption patterns for a specific customer"""
        customer_data = df[df['customer_id'] == customer_id].sort_values('date')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Daily consumption
        axes[0].plot(customer_data['date'], customer_data['daily_consumption_kwh'], 
                    linewidth=1.5, color='steelblue')
        axes[0].axhline(y=customer_data['consumption_mean'].iloc[0], 
                       color='red', linestyle='--', label='Mean Consumption')
        axes[0].set_xlabel('Date', fontsize=11)
        axes[0].set_ylabel('Daily Consumption (kWh)', fontsize=11)
        axes[0].set_title(f'Consumption Pattern - Customer {customer_id} (NTL: {customer_data["is_ntl"].iloc[0]})', 
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Rolling statistics
        axes[1].plot(customer_data['date'], customer_data['consumption_rolling_mean_30d'], 
                    label='30-day Rolling Mean', linewidth=2, color='green')
        axes[1].fill_between(customer_data['date'], 
                            customer_data['consumption_rolling_mean_30d'] - customer_data['consumption_rolling_std_30d'],
                            customer_data['consumption_rolling_mean_30d'] + customer_data['consumption_rolling_std_30d'],
                            alpha=0.3, color='green', label='±1 Std Dev')
        axes[1].set_xlabel('Date', fontsize=11)
        axes[1].set_ylabel('Consumption (kWh)', fontsize=11)
        axes[1].set_title('Rolling Statistics (30-day window)', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importance_df, top_n=15):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance for NTL Detection', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================================
# PART 5: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("="*60)
    print("NON-TECHNICAL LOSSES DETECTION SYSTEM")
    print("Power Distribution Network Analysis")
    print("="*60)
    
    # ========================================
    # Step 1: Data Simulation
    # ========================================
    print("\n[STEP 1] Simulating Power Consumption Data...")
    simulator = PowerConsumptionSimulator(n_customers=1000, n_days=365)
    df_normal = simulator.generate_normal_consumption(customer_type='residential')
    df_with_ntl = simulator.inject_ntl_patterns(df_normal, ntl_ratio=0.15)
    
    print(f"  - Total Records: {len(df_with_ntl):,}")
    print(f"  - Unique Customers: {df_with_ntl['customer_id'].nunique()}")
    print(f"  - NTL Cases: {df_with_ntl['is_ntl'].sum():,} ({df_with_ntl['is_ntl'].mean()*100:.2f}%)")
    
    # ========================================
    # Step 2: Feature Engineering
    # ========================================
    print("\n[STEP 2] Engineering Features...")
    engineer = NTLFeatureEngineer()
    df_features = engineer.engineer_features(df_with_ntl)
    
    feature_cols = [
        'daily_consumption_kwh', 'peak_hour_consumption', 'day_of_week', 'month',
        'consumption_mean', 'consumption_std', 'consumption_min', 'consumption_max',
        'consumption_deviation', 'consumption_rolling_mean_30d', 'consumption_rolling_std_30d',
        'consumption_diff', 'consumption_pct_change', 'sudden_drop', 'cv', 'ratio_to_max'
    ]
    
    print(f"  - Features Created: {len(feature_cols)}")
    
    # ========================================
    # Step 3: Train-Test Split
    # ========================================
    print("\n[STEP 3] Splitting Data...")
    X, y = df_features[feature_cols].values, df_features['is_ntl'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"  - Training Set: {len(X_train):,} samples")
    print(f"  - Test Set: {len(X_test):,} samples")
    
    # ========================================
    # Step 4: Model Training
    # ========================================
    print("\n[STEP 4] Training Detection Models...")
    detector = NTLDetector()
    detector.feature_columns = feature_cols
    
    # Train all models
    detector.train_supervised_models(X_train, y_train)
    detector.train_unsupervised_models(X_train)
    detector.train_statistical_detector(X_train)
    
    # ========================================
    # Step 5: Evaluation
    # ========================================
    print("\n[STEP 5] Evaluating Models...")
    
    models_to_evaluate = ['random_forest', 'gradient_boosting', 'isolation_forest', 
                          'one_class_svm', 'statistical']
    
    results = {}
    evaluator = NTLEvaluator()
    
    for model_name in models_to_evaluate:
        y_pred, y_prob = detector.predict(X_test, model_name)
        results[model_name] = (y_pred, y_prob)
        evaluator.evaluate_model(y_test, y_pred, y_prob, model_name.upper())
    
    # ========================================
    # Step 6: Visualizations
    # ========================================
    print("\n[STEP 6] Generating Visualizations...")
    
    # ROC Curves
    evaluator.plot_roc_curves(results, y_test)
    
    # Feature Importance
    importance_df = detector.get_feature_importance('random_forest')
    if importance_df is not None:
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        evaluator.plot_feature_importance(importance_df)
    
    # Example consumption patterns
    ntl_customer = df_features[df_features['is_ntl'] == 1]['customer_id'].iloc[0]
    normal_customer = df_features[df_features['is_ntl'] == 0]['customer_id'].iloc[0]
    
    print(f"\nVisualizing NTL Customer: {ntl_customer}")
    evaluator.plot_consumption_patterns(df_features, ntl_customer)
    
    print(f"\nVisualizing Normal Customer: {normal_customer}")
    evaluator.plot_consumption_patterns(df_features, normal_customer)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return df_features, detector, results


# ============================================================================
# RUN THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    df_features, detector, results = main()
    
    print("\n" + "="*60)
    print("SYSTEM READY FOR PRODUCTION USE")
    print("="*60)
    print("\nAvailable objects:")
    print("  - df_features: Full dataset with engineered features")
    print("  - detector: Trained NTL detection models")
    print("  - results: Dictionary of model predictions")
    print("\nFor new predictions, use:")
    print("  y_pred, y_prob = detector.predict(X_new, 'random_forest')")
