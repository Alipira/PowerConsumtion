"""
Enhanced Power Consumption Simulator with NTL Detection
========================================================

Mathematical Foundations:
------------------------

1. TEMPORAL PATTERN GENERATION:
   P(t) = P_base × M_hourly(h) × M_seasonal(s) × M_weekday(w) + ε(t)
   
   where:
   - P(t): Power consumption at time t
   - P_base ~ N(μ_type, σ_type): Base consumption from normal distribution
   - M_hourly(h): Hourly multiplier for hour h ∈ [0, 23]
   - M_seasonal(s): Seasonal multiplier for season s ∈ {winter, spring, summer, autumn}
   - M_weekday(w): Weekday/weekend multiplier
   - ε(t) ~ N(0, σ_noise): Gaussian noise

2. CYCLICAL FEATURE ENCODING:
   hour_sin = sin(2π × hour / 24)
   hour_cos = cos(2π × hour / 24)
   
   This preserves cyclical continuity (hour 23 is close to hour 0)

3. POWER FACTOR CALCULATION:
   PF(t) = {
       1.1 × L(t) - 0.05,     if L(t) < 0.5
       1.05 × L(t) - 0.025,   if L(t) ≥ 0.5
   }
   
   where L(t) = P(t) / max_t P(t) is the load ratio

4. ELECTRICAL METRICS:
   S(t) = P(t) / PF(t)                    [Apparent Power, kVA]
   I(t) = S(t) × 1000 / V                  [Current, A]
   Q(t) = √(S²(t) - P²(t))                 [Reactive Power, kVAR]

5. NTL FRAUD PATTERNS:
   
   a) Meter Bypass (Zero Consumption):
      P_fraud(t) = α × P_normal(t),  α ∈ [0, 0.3]
   
   b) Meter Tampering (Reduced Reading):
      P_fraud(t) = β × P_normal(t),  β ∈ [0.3, 0.7]
   
   c) Gradual Theft (Slow Decrease):
      P_fraud(t) = P_normal(t) × (1 - γt/T),  γ ∈ [0.1, 0.5]
   
   d) Periodic Theft (Time-based):
      P_fraud(t) = {
          δ × P_normal(t),  if t ∈ theft_periods
          P_normal(t),       otherwise
      }

6. ANOMALY DETECTION SCORES:
   
   a) Statistical Score:
      S_stat = |P_observed - E[P]| / σ[P]
      
   b) Pattern Deviation Score (Euclidean Distance):
      S_pattern = ||X_observed - X_cluster_center||_2
      
   c) Expected vs Actual Score:
      S_expected = |P_actual - P_expected| / P_expected
      
      where P_expected = (Total_Transformer / ΣP_installed) × P_i
   
   d) Composite Score:
      S_final = w₁ × S_stat + w₂ × S_pattern + w₃ × S_expected
      
      Typical weights: w₁ = 0.3, w₂ = 0.5, w₃ = 0.2

7. RECURRENCE PLOT TRANSFORMATION:
   R_i,j = {
       1,  if d(x(i), x(j)) < ε
       0,  otherwise
   }
   
   Used for converting time series to images for CNN-based detection
"""

from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


class FraudType(Enum):
    """Types of NTL fraud patterns"""
    NONE = "none"
    METER_BYPASS = "meter_bypass"
    METER_TAMPERING = "meter_tampering"
    GRADUAL_THEFT = "gradual_theft"
    PERIODIC_THEFT = "periodic_theft"


@dataclass
class CustomerTypeConfig:
    """Configuration for customer type consumption patterns"""
    mean_kwh: float
    std_kwh: float
    hourly_pattern: List[float]
    seasonal_factors: Dict[str, float]
    weekday_factor: float
    weekend_factor: float


@dataclass
class FraudConfig:
    """Configuration for fraud injection"""
    fraud_type: FraudType
    reduction_factor: float = 0.5  # α, β in formulas
    gradual_rate: float = 0.3      # γ in formula
    theft_hours: List[int] = None  # For periodic theft


class EnhancedPowerSimulator:
    """
    Enhanced power consumption simulator with NTL detection capabilities.
    
    Features:
    - Realistic consumption patterns with temporal variations
    - Fraud injection with multiple patterns
    - Anomaly detection and scoring
    - Time series to image conversion (Recurrence Plots)
    - Statistical feature extraction
    """
    
    CUSTOMER_TYPES = Literal['residential', 'commercial', 'industrial']
    SEASONS = ['winter', 'spring', 'summer', 'autumn']
    HOURS_PER_DAY = 24
    
    def __init__(
        self,
        n_customers: int,
        n_days: int = 365,
        start_date: str = '2023-01-01',
        random_seed: Optional[int] = None,
        fraud_ratio: float = 0.04
    ):
        """
        Initialize the enhanced power consumption simulator.
        
        Args:
            n_customers: Number of customers to simulate
            n_days: Number of days to simulate
            start_date: Start date for simulation
            random_seed: Random seed for reproducibility
            fraud_ratio: Proportion of fraudulent customers (default 4% as per research)
        """
        if n_customers <= 0:
            raise ValueError("n_customers must be positive")
        if n_days <= 0:
            raise ValueError("n_days must be positive")
        if not 0 <= fraud_ratio <= 1:
            raise ValueError("fraud_ratio must be between 0 and 1")
            
        self.n_customers = n_customers
        self.n_days = n_days
        self.start_date = pd.to_datetime(start_date)
        self.fraud_ratio = fraud_ratio
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self._customer_configs = self._initialize_customer_configs()
        self._cached_consumption_df: Optional[pd.DataFrame] = None
        self._fraud_labels: Optional[pd.Series] = None
        
    def _initialize_customer_configs(self) -> Dict[str, CustomerTypeConfig]:
        """Initialize consumption configurations for each customer type"""
        return {
            'residential': CustomerTypeConfig(
                mean_kwh=329, std_kwh=450,
                hourly_pattern=[
                    0.60, 0.55, 0.50, 0.50, 0.55, 0.70, 1.00, 1.20,
                    1.10, 0.90, 0.85, 0.85, 0.90, 0.90, 0.85, 0.90,
                    1.00, 1.30, 1.50, 1.60, 1.50, 1.30, 1.00, 0.80
                ],
                seasonal_factors={'winter': 1.15, 'spring': 0.95, 'summer': 1.25, 'autumn': 1.00},
                weekday_factor=1.00, weekend_factor=1.10
            ),
            'commercial': CustomerTypeConfig(
                mean_kwh=956, std_kwh=1200,
                hourly_pattern=[
                    0.20, 0.18, 0.15, 0.15, 0.18, 0.30, 0.40, 0.80,
                    1.00, 1.10, 1.05, 1.00, 0.95, 0.95, 1.00, 1.05,
                    1.00, 0.80, 0.50, 0.30, 0.25, 0.22, 0.20, 0.20
                ],
                seasonal_factors={'winter': 1.05, 'spring': 0.95, 'summer': 1.10, 'autumn': 1.00},
                weekday_factor=1.00, weekend_factor=0.70
            ),
            'industrial': CustomerTypeConfig(
                mean_kwh=2966, std_kwh=4500,
                hourly_pattern=[
                    0.85, 0.85, 0.85, 0.85, 0.85, 0.90, 0.95, 1.00,
                    1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05,
                    1.05, 1.00, 0.98, 0.95, 0.92, 0.90, 0.90, 0.88
                ],
                seasonal_factors={'winter': 1.02, 'spring': 0.98, 'summer': 1.03, 'autumn': 0.97},
                weekday_factor=1.00, weekend_factor=0.95
            )
        }
    
    def _get_season(self, day: int) -> str:
        """Determine season based on day of year"""
        season_idx = (day // 90) % 4
        return self.SEASONS[season_idx]
    
    def generate_consumption_data(
        self,
        customer_distribution: Tuple[float, float, float] = (0.6, 0.25, 0.15),
        noise_level: float = 0.05,
        inject_fraud: bool = True
    ) -> pd.DataFrame:
        """
        Generate power consumption data with optional fraud injection.
        
        Mathematical Formula:
        P(t) = P_base × M_hourly(h) × M_seasonal(s) × M_weekday(w) + ε(t)
        
        Args:
            customer_distribution: Distribution of (residential, commercial, industrial)
            noise_level: Noise proportion (default 5%)
            inject_fraud: Whether to inject fraud patterns
            
        Returns:
            DataFrame with consumption data and fraud labels
        """
        if not np.isclose(sum(customer_distribution), 1.0):
            raise ValueError("Distribution probabilities must sum to 1.0")
        
        # Generate customer assignments
        customer_ids = [f'customer_{i+1:04d}' for i in range(self.n_customers)]
        customer_types = np.random.choice(
            list(self._customer_configs.keys()),
            size=self.n_customers,
            p=customer_distribution
        )
        
        # Determine fraudulent customers
        n_fraud = int(self.n_customers * self.fraud_ratio)
        fraud_indices = np.random.choice(self.n_customers, size=n_fraud, replace=False)
        fraud_customers = set(fraud_indices)
        
        # Create datetime index
        datetime_index = pd.date_range(
            start=self.start_date,
            periods=self.n_days * self.HOURS_PER_DAY,
            freq='h'
        )
        
        # Pre-compute temporal features
        temporal_features = self._compute_temporal_features(datetime_index)
        
        # Generate consumption for each customer
        consumption_records = []
        fraud_labels = []
        
        for idx, (customer_id, customer_type) in enumerate(zip(customer_ids, customer_types)):
            config = self._customer_configs[customer_type]
            
            # Generate base consumption: P_base ~ N(μ_type, σ_type)
            base_consumption = np.maximum(
                np.random.normal(config.mean_kwh, config.std_kwh, len(datetime_index)),
                0.01
            )
            
            # Apply temporal patterns
            consumption = self._apply_temporal_patterns(
                base_consumption, temporal_features, config
            )
            
            # Add noise: ε(t) ~ N(0, σ_noise)
            noise = np.random.normal(0, noise_level * consumption)
            consumption = np.maximum(consumption + noise, 0)
            
            # Inject fraud if customer is fraudulent
            is_fraud = idx in fraud_customers
            if inject_fraud and is_fraud:
                fraud_type = np.random.choice(list(FraudType)[1:])  # Exclude NONE
                consumption = self._inject_fraud(consumption, fraud_type, datetime_index)
            
            # Create records
            for timestamp, kwh in zip(datetime_index, consumption):
                consumption_records.append({
                    'timestamp': timestamp,
                    'customer_id': customer_id,
                    'customer_type': customer_type,
                    'consumption_kwh': kwh,
                    'is_fraud': is_fraud
                })
            
            fraud_labels.extend([is_fraud] * len(datetime_index))
        
        # Create DataFrame
        df = pd.DataFrame(consumption_records)
        df = df.set_index('timestamp').sort_index()
        
        # Cache results
        self._cached_consumption_df = df
        self._fraud_labels = pd.Series(fraud_labels, index=df.index)
        
        return df
    
    def _compute_temporal_features(
        self,
        datetime_index: pd.DatetimeIndex,
        include_cyclical: bool = True
    ) -> pd.DataFrame:
        """
        Extract temporal features with cyclical encoding.
        
        Cyclical Encoding Formula:
        feature_sin = sin(2π × feature / period)
        feature_cos = cos(2π × feature / period)
        """
        features = pd.DataFrame({
            'hour': datetime_index.hour,
            'day_of_week': datetime_index.dayofweek,
            'day_of_year': datetime_index.dayofyear,
            'is_weekend': datetime_index.dayofweek.isin([5, 6])
        }, index=datetime_index)
        
        if include_cyclical:
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            features['day_of_year_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
            features['day_of_year_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
        
        return features
    
    def _apply_temporal_patterns(
        self,
        base_consumption: np.ndarray,
        temporal_features: pd.DataFrame,
        config: CustomerTypeConfig
    ) -> np.ndarray:
        """
        Apply temporal patterns to base consumption.
        
        Formula: P(t) = P_base × M_hourly × M_seasonal × M_weekday
        """
        # Hourly pattern
        hourly_multiplier = np.array([
            config.hourly_pattern[hour] for hour in temporal_features['hour']
        ])
        
        # Seasonal pattern
        seasonal_multiplier = np.array([
            config.seasonal_factors[self._get_season(day)]
            for day in temporal_features['day_of_year']
        ])
        
        # Weekday/weekend pattern
        weekday_multiplier = np.where(
            temporal_features['is_weekend'],
            config.weekend_factor,
            config.weekday_factor
        )
        
        # Combined pattern
        total_multiplier = hourly_multiplier * seasonal_multiplier * weekday_multiplier
        
        return base_consumption * total_multiplier
    
    def _inject_fraud(
        self,
        normal_consumption: np.ndarray,
        fraud_type: FraudType,
        datetime_index: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Inject fraud patterns into consumption data.
        
        Fraud Formulas:
        1. Meter Bypass: P_fraud = α × P_normal, α ∈ [0, 0.3]
        2. Meter Tampering: P_fraud = β × P_normal, β ∈ [0.3, 0.7]
        3. Gradual Theft: P_fraud = P_normal × (1 - γt/T)
        4. Periodic Theft: P_fraud = δ × P_normal (during theft periods)
        """
        fraud_consumption = normal_consumption.copy()
        
        if fraud_type == FraudType.METER_BYPASS:
            # Near-zero consumption: α ∈ [0, 0.3]
            reduction = np.random.uniform(0, 0.3)
            fraud_consumption = reduction * normal_consumption
            
        elif fraud_type == FraudType.METER_TAMPERING:
            # Reduced reading: β ∈ [0.3, 0.7]
            reduction = np.random.uniform(0.3, 0.7)
            fraud_consumption = reduction * normal_consumption
            
        elif fraud_type == FraudType.GRADUAL_THEFT:
            # Gradual decrease: P_fraud = P_normal × (1 - γt/T)
            gamma = np.random.uniform(0.1, 0.5)
            T = len(normal_consumption)
            decline = np.linspace(0, gamma, T)
            fraud_consumption = normal_consumption * (1 - decline)
            
        elif fraud_type == FraudType.PERIODIC_THEFT:
            # Theft during specific hours: δ ∈ [0.2, 0.6]
            theft_hours = np.random.choice(24, size=np.random.randint(4, 12), replace=False)
            reduction = np.random.uniform(0.2, 0.6)
            hours = datetime_index.hour
            mask = np.isin(hours, theft_hours)
            fraud_consumption[mask] = reduction * normal_consumption[mask]
        
        return np.maximum(fraud_consumption, 0)
    
    def calculate_fraud_scores(
        self,
        df: Optional[pd.DataFrame] = None,
        weights: Tuple[float, float, float] = (0.3, 0.5, 0.2)
    ) -> pd.DataFrame:
        """
        Calculate fraud detection scores for each customer.
        
        Composite Score Formula:
        S_final = w₁ × S_stat + w₂ × S_pattern + w₃ × S_expected
        
        Args:
            df: Consumption DataFrame (uses cached if None)
            weights: Weights for (statistical, pattern, expected) scores
            
        Returns:
            DataFrame with fraud scores per customer
        """
        if df is None:
            if self._cached_consumption_df is None:
                raise ValueError("No consumption data. Run generate_consumption_data() first.")
            df = self._cached_consumption_df
        
        # Calculate component scores
        stat_scores = self._calculate_statistical_score(df)
        pattern_scores = self._calculate_pattern_score(df)
        expected_scores = self._calculate_expected_score(df)
        
        # Composite score: S_final = w₁×S_stat + w₂×S_pattern + w₃×S_expected
        w1, w2, w3 = weights
        final_scores = w1 * stat_scores + w2 * pattern_scores + w3 * expected_scores
        
        scores_df = pd.DataFrame({
            'customer_id': stat_scores.index,
            'statistical_score': stat_scores.values,
            'pattern_score': pattern_scores.values,
            'expected_score': expected_scores.values,
            'final_score': final_scores.values,
            'is_fraud': df.groupby('customer_id')['is_fraud'].first().values
        })
        
        return scores_df.sort_values('final_score', ascending=False)
    
    def _calculate_statistical_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate statistical anomaly score.
        
        Formula: S_stat = |P_observed - E[P]| / σ[P]
        """
        customer_stats = df.groupby('customer_id')['consumption_kwh'].agg(['mean', 'std'])
        customer_recent = df.groupby('customer_id')['consumption_kwh'].tail(24).groupby('customer_id').mean()
        
        # Z-score of recent consumption vs historical
        z_scores = np.abs(customer_recent - customer_stats['mean']) / (customer_stats['std'] + 1e-6)
        
        # Normalize to [0, 1]
        normalized = (z_scores - z_scores.min()) / (z_scores.max() - z_scores.min() + 1e-6)
        
        return normalized
    
    def _calculate_pattern_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate pattern deviation score using consumption profile changes.
        
        Formula: S_pattern = ||X_observed - X_expected||₂
        """
        scores = []
        customer_ids = []
        
        for customer_id in df['customer_id'].unique():
            customer_data = df[df['customer_id'] == customer_id]['consumption_kwh'].values
            
            # Split into first half and second half
            mid = len(customer_data) // 2
            first_half = customer_data[:mid]
            second_half = customer_data[mid:]
            
            # Calculate statistics for each half
            first_stats = np.array([np.mean(first_half), np.std(first_half), np.max(first_half)])
            second_stats = np.array([np.mean(second_half), np.std(second_half), np.max(second_half)])
            
            # Euclidean distance between patterns
            pattern_distance = euclidean(first_stats, second_stats)
            
            scores.append(pattern_distance)
            customer_ids.append(customer_id)
        
        scores = np.array(scores)
        # Normalize to [0, 1]
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        return pd.Series(normalized, index=customer_ids)
    
    def _calculate_expected_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate expected vs actual consumption score.
        
        Formula: S_expected = |P_actual - P_expected| / P_expected
        where P_expected = (Total_Transformer / ΣP_installed) × P_i
        """
        # Simulate installed power (proportional to mean consumption)
        customer_means = df.groupby('customer_id')['consumption_kwh'].mean()
        installed_power = customer_means * np.random.uniform(1.5, 2.5, size=len(customer_means))
        
        # Calculate expected consumption
        total_consumption = customer_means.sum()
        total_installed = installed_power.sum()
        
        expected_consumption = (total_consumption / total_installed) * installed_power
        
        # Score: |actual - expected| / expected
        scores = np.abs(customer_means - expected_consumption) / (expected_consumption + 1e-6)
        
        # Normalize to [0, 1]
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        return normalized
    
    def create_recurrence_plot(
        self,
        time_series: np.ndarray,
        epsilon: Optional[float] = None
    ) -> np.ndarray:
        """
        Create recurrence plot from time series.
        
        Formula:
        R_i,j = 1 if d(x(i), x(j)) < ε, else 0
        
        Args:
            time_series: 1D time series data
            epsilon: Distance threshold (auto-calculated if None)
            
        Returns:
            2D recurrence plot matrix
        """
        N = len(time_series)
        
        # Auto-calculate epsilon as 10% of std
        if epsilon is None:
            epsilon = 0.1 * np.std(time_series)
        
        # Calculate pairwise distances
        recurrence_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                distance = np.abs(time_series[i] - time_series[j])
                recurrence_matrix[i, j] = 1 if distance < epsilon else 0
        
        return recurrence_matrix
    
    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive statistical features per customer.
        
        Features:
        - Mean, Median, Std, Min, Max, Quantiles
        - Coefficient of Variation
        - Skewness, Kurtosis
        - Trend (linear regression slope)
        """
        from scipy.stats import skew, kurtosis, linregress
        
        features_list = []
        
        for customer_id in df['customer_id'].unique():
            data = df[df['customer_id'] == customer_id]['consumption_kwh'].values
            
            # Basic statistics
            features = {
                'customer_id': customer_id,
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'cv': np.std(data) / (np.mean(data) + 1e-6),  # Coefficient of Variation
                'skewness': skew(data),
                'kurtosis': kurtosis(data),
            }
            
            # Trend (slope of linear regression)
            x = np.arange(len(data))
            slope, _, _, _, _ = linregress(x, data)
            features['trend'] = slope
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)


# Example Usage
if __name__ == "__main__":
    print("="*80)
    print("ENHANCED POWER CONSUMPTION SIMULATOR WITH NTL DETECTION")
    print("="*80)
    
    # Create simulator
    simulator = EnhancedPowerSimulator(
        n_customers=100,
        n_days=90,
        random_seed=42,
        fraud_ratio=0.04
    )
    
    # Generate data with fraud injection
    print("\n1. Generating consumption data with fraud patterns...")
    consumption_df = simulator.generate_consumption_data(inject_fraud=True)
    
    print(f"   Generated {len(consumption_df)} records")
    print(f"   Fraud ratio: {consumption_df['is_fraud'].mean():.2%}")
    print(f"   Data shape: {consumption_df.shape}")
    
    # Calculate fraud scores
    print("\n2. Calculating fraud detection scores...")
    scores_df = simulator.calculate_fraud_scores()
    
    print("\n   Top 10 Suspicious Customers:")
    print(scores_df.head(10).to_string(index=False))
    
    # Extract statistical features
    print("\n3. Extracting statistical features...")
    features_df = simulator.extract_statistical_features(consumption_df)
    print(f"   Extracted {features_df.shape[1]-1} features for {features_df.shape[0]} customers")
    
    # Performance metrics
    from sklearn.metrics import classification_report, roc_auc_score
    
    # Use final score as classifier (threshold = 0.6)
    threshold = 0.6
    predicted_fraud = (scores_df['final_score'] > threshold).astype(int)
    actual_fraud = scores_df['is_fraud'].astype(int)
    
    print("\n4. Detection Performance (threshold = 0.6):")
    print(classification_report(actual_fraud, predicted_fraud, 
                                target_names=['Normal', 'Fraud']))
    
    # Calculate AUC-ROC
    auc_score = roc_auc_score(actual_fraud, scores_df['final_score'])
    print(f"\n   AUC-ROC Score: {auc_score:.4f}")
    
    # Example: Create recurrence plot for one customer
    print("\n5. Creating recurrence plot for sample customer...")
    sample_customer = consumption_df[consumption_df['customer_id'] == 'customer_0001']
    sample_series = sample_customer['consumption_kwh'].values[:168]  # 1 week
    
    rp = simulator.create_recurrence_plot(sample_series)
    print(f"   Recurrence plot shape: {rp.shape}")
    print(f"   Recurrence rate: {rp.mean():.2%}")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
