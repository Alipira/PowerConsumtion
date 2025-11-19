"""
Power Consumption Simulator for Time Series Analysis

This module provides a simulator for generating realistic power consumption
data across different customer types with temporal and seasonal variations.
Then inject NTL into the data
"""

from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CustomerTypeConfig:
    """Configuration for a customer type's consumption patterns."""
    mean_kwh: float
    std_kwh: float
    hourly_pattern: List[float]
    seasonal_factors: Dict[str, float]
    weekday_factor: float
    weekend_factor: float


class PowerConsumptionSimulator:
    """
    Simulates power consumption data for multiple customers over time.

    Generates realistic consumption patterns considering:
    - Customer type (residential, commercial, industrial)
    - Hourly variations
    - Seasonal variations
    - Weekday/weekend patterns
    """

    CUSTOMER_TYPES = Literal['residential', 'commercial', 'industrial']
    SEASONS = ['winter', 'spring', 'summer', 'autumn']
    HOURS_PER_DAY = 24

    def __init__(
        self,
        n_customers: int,
        n_days: int = 365,
        start_date: str = '2023-01-01',
        random_seed: Optional[int] = None
    ):
        """
        Initialize the power consumption simulator.

        Args:
            n_customers: Number of customers to simulate
            n_days: Number of days to simulate
            start_date: Start date for the simulation
            random_seed: Random seed for reproducibility
        """
        if n_customers <= 0:
            raise ValueError("n_customers must be positive")
        if n_days <= 0:
            raise ValueError("n_days must be positive")

        self.n_customers = n_customers
        self.n_days = n_days
        self.start_date = pd.to_datetime(start_date)

        if random_seed is not None:
            np.random.seed(random_seed)

        self._customer_configs = self._initialize_customer_configs()
        self._cached_consumption_df: Optional[pd.DataFrame] = None

    def _initialize_customer_configs(self) -> Dict[str, CustomerTypeConfig]:
        """Initialize consumption configurations for each customer type."""
        return {
            'residential': CustomerTypeConfig(
                mean_kwh=329,
                std_kwh=450,
                hourly_pattern=[
                    0.60, 0.55, 0.50, 0.50, 0.55, 0.70, 1.00, 1.20,
                    1.10, 0.90, 0.85, 0.85, 0.90, 0.90, 0.85, 0.90,
                    1.00, 1.30, 1.50, 1.60, 1.50, 1.30, 1.00, 0.80
                ],
                seasonal_factors={
                    'winter': 1.15, 'spring': 0.95,
                    'summer': 1.25, 'autumn': 1.00
                },
                weekday_factor=1.00,
                weekend_factor=1.10
            ),
            'commercial': CustomerTypeConfig(
                mean_kwh=956,
                std_kwh=1200,
                hourly_pattern=[
                    0.20, 0.18, 0.15, 0.15, 0.18, 0.30, 0.40, 0.80,
                    1.00, 1.10, 1.05, 1.00, 0.95, 0.95, 1.00, 1.05,
                    1.00, 0.80, 0.50, 0.30, 0.25, 0.22, 0.20, 0.20
                ],
                seasonal_factors={
                    'winter': 1.05, 'spring': 0.95,
                    'summer': 1.10, 'autumn': 1.00
                },
                weekday_factor=1.00,
                weekend_factor=0.70
            ),
            'industrial': CustomerTypeConfig(
                mean_kwh=2966,
                std_kwh=4500,
                hourly_pattern=[
                    0.85, 0.85, 0.85, 0.85, 0.85, 0.90, 0.95, 1.00,
                    1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05,
                    1.05, 1.00, 0.98, 0.95, 0.92, 0.90, 0.90, 0.88
                ],
                seasonal_factors={
                    'winter': 1.02, 'spring': 0.98,
                    'summer': 1.03, 'autumn': 0.97
                },
                weekday_factor=1.00,
                weekend_factor=0.95
            )
        }

    def _get_season(self, day: int) -> str:
        """Determine season based on day of year."""
        season_idx = (day // 90) % 4
        return self.SEASONS[season_idx]

    def _generate_customer_assignments(
        self,
        distribution: Tuple[float, float, float] = (0.6, 0.25, 0.15)
    ) -> pd.DataFrame:
        """
        Generate customer IDs and type assignments.

        Args:
            distribution: Probability distribution for (residential, commercial, industrial)

        Returns:
            DataFrame with customer_id and customer_type
        """
        if not np.isclose(sum(distribution), 1.0):
            raise ValueError("Distribution probabilities must sum to 1.0")

        customer_ids = [f'customer_{i+1:04d}' for i in range(self.n_customers)]
        customer_types = np.random.choice(
            list(self._customer_configs.keys()),
            size=self.n_customers,
            p=distribution
        )

        return pd.DataFrame({
            'customer_id': customer_ids,
            'customer_type': customer_types
        })

    def generate_consumption_data(
        self,
        customer_distribution: Tuple[float, float, float] = (0.6, 0.25, 0.15),
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate power consumption data for all customers.

        Args:
            customer_distribution: Distribution of customer types
            noise_level: Proportion of noise to add (default 5%)

        Returns:
            DataFrame with timestamp index and consumption data
        """
        # Generate customer assignments
        customers_df = self._generate_customer_assignments(customer_distribution)

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

        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            customer_type = customer['customer_type']
            config = self._customer_configs[customer_type]

            # Generate base consumption
            base_consumption = np.maximum(
                np.random.normal(config.mean_kwh, config.std_kwh, len(datetime_index)),
                0.01
            )

            # Apply temporal patterns
            consumption = self._apply_temporal_patterns(
                base_consumption,
                temporal_features,
                config
            )

            # Add noise
            noise = np.random.normal(0, noise_level * consumption)
            consumption = np.maximum(consumption + noise, 0)

            # Create records
            for _, (timestamp, kwh) in enumerate(zip(datetime_index, consumption)):
                consumption_records.append({
                    'timestamp': timestamp,
                    'customer_id': customer_id,
                    'customer_type': customer_type,
                    'consumption_kwh': kwh
                })

        # Create DataFrame
        df = pd.DataFrame(consumption_records)
        df = df.set_index('timestamp').sort_index()

        # Cache for later use
        self._cached_consumption_df = df

        return df

    def _compute_temporal_features(self, datetime_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Extract temporal features from datetime index."""
        return pd.DataFrame({
            'hour': datetime_index.hour,
            'day_of_week': datetime_index.dayofweek,
            'day_of_year': datetime_index.dayofyear,
            'is_weekend': datetime_index.dayofweek.isin([6, 7])
        }, index=datetime_index)

    def _apply_temporal_patterns(
        self,
        base_consumption: np.ndarray,
        temporal_features: pd.DataFrame,
        config: CustomerTypeConfig
    ) -> np.ndarray:
        """Apply hourly, seasonal, and weekday/weekend patterns."""
        # Hourly pattern
        hourly_multiplier = np.array([
            config.hourly_pattern[hour]
            for hour in temporal_features['hour']
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

        # Combine all patterns
        total_multiplier = hourly_multiplier * seasonal_multiplier * weekday_multiplier

        return base_consumption * total_multiplier

    def calculate_power_factor(
        self,
        df: Optional[pd.DataFrame] = None,
        base_factor: float = 0.9
    ) -> pd.DataFrame:
        """
        Calculate variable power factor based on load ratio.

        Args:
            df: Consumption DataFrame (uses cached if None)
            base_factor: Base power factor when no load

        Returns:
            DataFrame with load_ratio and power_factor columns
        """
        if df is None:
            if self._cached_consumption_df is None:
                raise ValueError("No consumption data available. Run generate_consumption_data() first.")
            df = self._cached_consumption_df.copy()
        else:
            df = df.copy()

        # Validate inputs
        if 'customer_id' not in df.columns or 'consumption_kwh' not in df.columns:
            raise KeyError("DataFrame must contain 'customer_id' and 'consumption_kwh' columns")
        if not (0 < base_factor <= 1):
            raise ValueError("base_factor must be between 0 and 1")

        # Calculate max consumption per customer
        max_consumption = df.groupby('customer_id')['consumption_kwh'].transform('max')

        # Calculate load ratio (handle zero max)
        df['load_ratio'] = np.where(
            max_consumption > 0,
            df['consumption_kwh'] / max_consumption,
            np.nan
        )

        # Calculate power factor using piecewise function
        load_ratio = df['load_ratio'].values
        power_factor = np.where(
            np.isnan(load_ratio),
            base_factor,
            np.where(
                load_ratio < 0.5,
                load_ratio - 0.1 * (0.5 - load_ratio),
                load_ratio + 0.05 * (load_ratio - 0.5)
            )
        )

        # Clip to valid range [0, 1]
        df['power_factor'] = np.clip(power_factor, 0.01, 1.0)

        return df

    def calculate_electrical_metrics(
        self,
        df: Optional[pd.DataFrame] = None,
        voltage: float = 220.0,
        base_power_factor: float = 0.9
    ) -> pd.DataFrame:
        """
        Calculate electrical metrics: apparent power, current, and reactive power.

        Args:
            df: Consumption DataFrame (uses cached if None)
            voltage: Supply voltage in volts
            base_power_factor: Base power factor

        Returns:
            DataFrame with all electrical metrics
        """
        if voltage <= 0:
            raise ValueError("voltage must be positive")

        # Calculate power factor
        df = self.calculate_power_factor(df, base_power_factor)

        # Apparent Power S (kVA)
        df['apparent_power_kva'] = df['consumption_kwh'] / df['power_factor']

        # Current I (A)
        df['current_a'] = (df['apparent_power_kva'] * 1000) / voltage

        # Reactive Power Q (kVAR)
        df['reactive_power_kvar'] = np.sqrt(
            np.maximum(
                df['apparent_power_kva']**2 - df['consumption_kwh']**2,
                0
            )
        )

        return df

    # FIXME: Add type hints for return types, and create a feature engineering module
    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics by customer type.

        Args:
            df: Consumption DataFrame

        Returns:
            DataFrame with summary statistics
        """
        return df.groupby('customer_type')['consumption_kwh'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)

    def resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample hourly data to daily totals.

        Args:
            df: Hourly consumption DataFrame

        Returns:
            DataFrame with daily totals
        """
        return df.groupby(['customer_id', 'customer_type', pd.Grouper(freq='D')]).agg({
            'consumption_kwh': 'sum'
        }).reset_index()
