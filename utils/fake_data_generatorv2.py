import numpy as np
import pandas as pd
import os

from datetime import datetime
from dotenv import load_dotenv


# Load .env file
load_dotenv()
csv_path = os.getenv('csv_path')

if csv_path is None:
    raise ValueError(f"Environment variable {csv_path} is not set !!!")

full_path = os.path.join(csv_path, 'smart_meter_data_with_ntl.csv')

# Hyperparameters
NUM_METERS = 1000  # Total meters (90% normal, 10% NTL)
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
HOURS = int((END_DATE - START_DATE).total_seconds() / 3600) + 24
NORMAL_RATIO = 0.9
PHASES = 3  # Three-phase system


def generate_load_profile(hours, daily_peak, base_load, weekend_boost=1.2):
    """Generate synthetic load profile with daily/weekly seasonality"""
    t = np.arange(hours)
    # Daily pattern (peaks at 8AM and 8PM)
    daily = (np.sin(t * 2 * np.pi / 24 - np.pi / 2) + 1) * 0.5
    daily_peaks = 0.7 * np.exp(-(t % 24 - 8)**2/8) + 0.7 * np.exp(-(t % 24 - 20)**2/8)

    # Weekly pattern (higher usage on weekends)
    is_weekend = ((t // 24) % 7 >= 5).astype(float)
    weekly = is_weekend * weekend_boost + (1 - is_weekend) * 1.0

    # Combine patterns with noise
    load = base_load + (daily * 0.3 + daily_peaks * 0.7) * daily_peak * weekly
    load *= np.random.lognormal(mean=0.0, sigma=0.05, size=hours)
    return np.clip(load, base_load, None)


def calculate_derived_metrics(P, V, PF):
    """Calculate electrical metrics from active power"""
    S = P / PF  # Apparent power (kVA)
    I = S * 1000 / (PHASES**0.5 * V)  # Current in Amps
    Q = S * np.sqrt(1 - PF**2)  # Reactive power (kVAR)
    return S, I, Q


def ntl_generator(df, meter_type):
    """Generate NTL patterns based on theft type"""
    df = df.copy()
    timestamps = df.index.values
    hours = timestamps.astype('datetime64[h]').astype(int) % 24

    if meter_type == "partial_bypass":
        # Night-time theft (10PM-5AM)
        theft_mask = (hours >= 22) | (hours < 5)
        df.loc[theft_mask, 'P_actual'] *= 1.5  # Increase actual usage during theft
        df.loc[theft_mask, 'P_reported'] = df.loc[theft_mask, 'P_actual'] * 0.4

    elif meter_type == "voltage_tamper":
        # Random voltage manipulation events
        theft_events = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
        df.loc[theft_events.astype(bool), 'V'] *= 0.85  # Voltage drop
        # Recalculate metrics with tampered voltage
        S, I, Q = calculate_derived_metrics(df['P_actual'], df['V'], df['PF'])
        df['I'] = I
        df['Q'] = Q

    elif meter_type == "current_tamper":
        # Continuous current underreporting
        df['I'] *= 0.65
        # Recalculate power with tampered current
        df['P_reported'] = PHASES**0.5 * df['V'] * df['I'] * df['PF'] / 1000

    # Update cumulative energy and reactive power
    df['ActiveEnergy'] = df['P_reported'].cumsum()
    df['ReactiveEnergy'] = df['Q'].cumsum()
    return df


# Generate base dataset
data = []
rng = np.random.default_rng(seed=42)

for meter_id in range(NUM_METERS):
    # Meter characteristics
    is_ntl = meter_id >= NUM_METERS * NORMAL_RATIO
    meter_type = "normal"
    base_load = rng.uniform(0.05, 0.2)
    daily_peak = rng.uniform(1.5, 5.0)
    pf = rng.uniform(0.85, 0.98)
    v_nominal = 220 + rng.normal(0, 5)

    # Generate timestamp index
    timestamps = pd.date_range(START_DATE, periods=HOURS, freq='H')

    # Generate base load profile
    P_actual = generate_load_profile(HOURS, daily_peak, base_load)
    P_reported = P_actual.copy()

    # Calculate electrical parameters
    V = v_nominal * (1 - 0.05 * P_actual / P_actual.max())  # Voltage drop during peaks
    S, I, Q = calculate_derived_metrics(P_actual, V, pf)

    # Create meter dataframe
    meter_df = pd.DataFrame({
        'timestamp': timestamps,
        'P_actual': P_actual,
        'P_reported': P_reported,
        'V': V,
        'PF': pf,
        'I': I,
        'Q': Q
    }).set_index('timestamp')

    meter_df['ActiveEnergy'] = meter_df['P_reported'].cumsum()
    meter_df['ReactiveEnergy'] = meter_df['Q'].cumsum()

    # Introduce NTL patterns
    if is_ntl:
        meter_type = rng.choice(["partial_bypass", "voltage_tamper", "current_tamper"])
        meter_df = ntl_generator(meter_df, meter_type)

    # Add metadata
    meter_df['MeterID'] = meter_id
    meter_df['NTL_Type'] = "normal" if not is_ntl else meter_type
    meter_df['Phase'] = PHASES
    data.append(meter_df)

# Combine all meters
df = pd.concat(data).reset_index()

# Rename columns
df = df.rename(columns={
    'ReactiveEnergy': 'ReactiveCumulativeEnergyImport(+R)',
    'ActiveEnergy': 'DemandRegisterActiveEnergyCombinedTotal',
    'P_reported': 'LastAverageValueOfImportActivePower',
    'PF': 'LastAverageValueOfPFTotal',
    'I': 'LastAverageValueOfCurrent',
    'V': 'LastAverageValueOfVoltageL'
})

# Add negative reactive energy (typically zero for residential)
df['ReactiveCumulativeEnergyImport(-R)'] = 0

# Final column ordering
cols = [
    'MeterID', 'timestamp', 'Phase', 'NTL_Type',
    'ReactiveCumulativeEnergyImport(+R)',
    'ReactiveCumulativeEnergyImport(-R)',
    'DemandRegisterActiveEnergyCombinedTotal',
    'LastAverageValueOfImportActivePower',
    'LastAverageValueOfPFTotal',
    'LastAverageValueOfCurrent',
    'LastAverageValueOfVoltageL'
]
df = df[cols]

# Save to CSV
df.to_csv(full_path, index=False)
