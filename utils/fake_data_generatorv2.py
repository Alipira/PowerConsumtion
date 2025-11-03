import numpy as np
import pandas as pd
import os
from datetime import datetime

csv_path = '/mnt/archive/project/PowerNTL/data/'

if csv_path is None:
    raise ValueError(f"Environment variable {csv_path} is not set !!!")

full_path = os.path.join(csv_path, 'smart_meter_data_with_ntl.csv')

# Hyperparameters
NUM_METERS = 1000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
HOURS = int((END_DATE - START_DATE).total_seconds() / 3600) + 24
NORMAL_RATIO = 0.9
NTL_RATIO = 1 - NORMAL_RATIO

# FIXED: Phase assignment based on customer type
RESIDENTIAL_RATIO = 0.7


def generate_load_profile(hours, daily_peak, base_load, weekend_boost=1.2, rng=None):
    """Generate synthetic load profile with daily/weekly/seasonal patterns"""
    if rng is None:
        rng = np.random.default_rng()
    
    t = np.arange(hours)
    
    # Daily pattern (peaks at 8AM and 8PM)
    daily = (np.sin(t * 2 * np.pi / 24 - np.pi / 2) + 1) * 0.5
    daily_peaks = 0.7 * np.exp(-(t % 24 - 8)**2/8) + 0.7 * np.exp(-(t % 24 - 20)**2/8)

    # Weekly pattern (higher usage on weekends)
    is_weekend = ((t // 24) % 7 >= 5).astype(float)
    weekly = is_weekend * weekend_boost + (1 - is_weekend) * 1.0

    # FIXED: Add seasonal variation (annual cycle)
    days = t / 24
    seasonal = 1 + 0.3 * np.sin(2 * np.pi * days / 365 - np.pi/2)  # Peak in summer

    # Combine patterns with noise
    load = base_load + (daily * 0.3 + daily_peaks * 0.7) * daily_peak * weekly * seasonal
    
    # FIXED: Use passed rng for reproducibility
    load *= rng.lognormal(mean=0.0, sigma=0.05, size=hours)
    return np.clip(load, base_load * 0.5, None)


def calculate_variable_pf(P, P_max, base_pf):
    """Calculate variable power factor based on load (more realistic)"""
    load_ratio = P / (P_max + 1e-6)
    # Power factor degrades at low loads
    pf_values = base_pf * (0.85 + 0.15 * load_ratio)
    return np.clip(pf_values, 0.7, 0.99)


def calculate_derived_metrics(P, V, PF, phases):
    """Calculate electrical metrics from active power"""
    S = P / PF  # Apparent power (kVA)
    I = S * 1000 / (phases**0.5 * V)  # Current in Amps (three-phase formula)
    Q = S * np.sqrt(np.clip(1 - PF**2, 0, 1))  # Reactive power (kVAR)
    return S, I, Q


def validate_data(df):
    """Validate physical constraints"""
    errors = []
    
    if (df['P_actual'] < 0).any():
        errors.append("Negative actual power detected")
    if (df['P_reported'] < 0).any():
        errors.append("Negative reported power detected")
    if not ((df['V'] > 180) & (df['V'] < 260)).all():
        errors.append(f"Voltage out of range: {df['V'].min():.1f} - {df['V'].max():.1f}V")
    if not ((df['PF'] >= 0.6) & (df['PF'] <= 1.0)).all():
        errors.append(f"Invalid PF: {df['PF'].min():.3f} - {df['PF'].max():.3f}")
    if (df['I'] < 0).any():
        errors.append("Negative current detected")
    
    if errors:
        print(f"⚠️  Validation warnings for Meter {df['MeterID'].iloc[0]}:")
        for err in errors:
            print(f"   - {err}")
    
    return df


def ntl_generator(df, meter_type, phases, rng):
    """Generate NTL patterns based on theft type"""
    df = df.copy()
    timestamps = df.index.values
    hours = timestamps.astype('datetime64[h]').astype(int) % 24

    if meter_type == "partial_bypass":
        # Night-time theft (10PM-5AM) - bypassing meter during off-peak
        theft_mask = (hours >= 22) | (hours < 5)
        df.loc[theft_mask, 'P_actual'] *= 1.5  # Actual usage increases
        df.loc[theft_mask, 'P_reported'] = df.loc[theft_mask, 'P_actual'] * 0.4  # Meter reports less

    elif meter_type == "voltage_tamper":
        # Random voltage manipulation events (affects all calculations)
        theft_events = rng.choice([0, 1], size=len(df), p=[0.95, 0.05])
        theft_mask = theft_events.astype(bool)
        
        # Reduce voltage artificially
        df.loc[theft_mask, 'V'] *= 0.85
        
        # FIXED: Recalculate all metrics with tampered voltage
        S, I, Q = calculate_derived_metrics(
            df.loc[theft_mask, 'P_actual'], 
            df.loc[theft_mask, 'V'], 
            df.loc[theft_mask, 'PF'],
            phases
        )
        df.loc[theft_mask, 'S'] = S
        df.loc[theft_mask, 'I'] = I
        df.loc[theft_mask, 'Q'] = Q
        # Reported power affected by voltage manipulation
        df.loc[theft_mask, 'P_reported'] = S * df.loc[theft_mask, 'PF']

    elif meter_type == "current_tamper":
        # FIXED: Direct tampering of reported power (meters measure power, not calculate it)
        # Continuous underreporting
        df['P_reported'] *= 0.65
        
        # Recalculate apparent current based on tampered reported power
        S_tampered = df['P_reported'] / df['PF']
        df['I'] = S_tampered * 1000 / (phases**0.5 * df['V'])
        df['S'] = S_tampered

    elif meter_type == "gradual_theft":
        # NEW: Gradually increasing theft over time
        theft_progression = np.linspace(1.0, 0.4, len(df))
        df['P_reported'] = df['P_actual'] * theft_progression

    elif meter_type == "intermittent":
        # NEW: Random days of theft
        days = len(df) // 24
        theft_days = rng.choice([0, 1], size=days, p=[0.7, 0.3])
        theft_mask = np.repeat(theft_days, 24)[:len(df)]
        df.loc[theft_mask.astype(bool), 'P_reported'] *= 0.5

    # Update cumulative energy
    df['ActiveEnergy'] = df['P_reported'].cumsum()
    df['ReactiveEnergy'] = df['Q'].cumsum()
    
    return df


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

print("="*70)
print("SMART METER DATA GENERATOR WITH NTL PATTERNS")
print("="*70)

# Generate base dataset
data = []
rng = np.random.default_rng(seed=42)

ntl_types = ["partial_bypass", "voltage_tamper", "current_tamper", "gradual_theft", "intermittent"]

for meter_id in range(NUM_METERS):
    if meter_id % 100 == 0:
        print(f"Generating meter {meter_id}/{NUM_METERS}...")
    
    # Meter characteristics
    is_residential = meter_id < NUM_METERS * RESIDENTIAL_RATIO
    is_ntl = meter_id >= NUM_METERS * NORMAL_RATIO
    
    # FIXED: Assign phases based on customer type
    phases = 1 if is_residential else 3
    
    meter_type = "normal"
    base_load = rng.uniform(0.05, 0.2) if is_residential else rng.uniform(0.5, 2.0)
    daily_peak = rng.uniform(1.5, 5.0) if is_residential else rng.uniform(10.0, 50.0)
    base_pf = rng.uniform(0.85, 0.98)
    v_nominal = 220 + rng.normal(0, 5)
    
    customer_type = 'Residential' if is_residential else 'Commercial'

    # Generate timestamp index
    timestamps = pd.date_range(START_DATE, periods=HOURS, freq='h')

    # Generate base load profile
    P_actual = generate_load_profile(HOURS, daily_peak, base_load, rng=rng)
    P_reported = P_actual.copy()

    # Calculate variable power factor
    PF = calculate_variable_pf(P_actual, P_actual.max(), base_pf)

    # Calculate electrical parameters
    V = v_nominal * (1 - 0.05 * P_actual / (P_actual.max() + 1e-6))  # Voltage drop during peaks
    S, I, Q = calculate_derived_metrics(P_actual, V, PF, phases)

    # Create meter dataframe
    meter_df = pd.DataFrame({
        'timestamp': timestamps,
        'P_actual': P_actual,
        'P_reported': P_reported,
        'V': V,
        'PF': PF,
        'I': I,
        'Q': Q,
        'S': S
    }).set_index('timestamp')

    meter_df['ActiveEnergy'] = meter_df['P_reported'].cumsum()
    meter_df['ReactiveEnergy'] = meter_df['Q'].cumsum()

    # Introduce NTL patterns
    if is_ntl:
        meter_type = rng.choice(ntl_types)
        meter_df = ntl_generator(meter_df, meter_type, phases, rng)

    # Add metadata
    meter_df['MeterID'] = meter_id
    meter_df['NTL_Type'] = meter_type
    meter_df['Phase'] = phases
    meter_df['CustomerType'] = customer_type
    meter_df['Region'] = rng.choice(['Urban', 'Suburban', 'Rural'])
    
    # FIXED: Validate data
    meter_df = validate_data(meter_df)
    
    data.append(meter_df)

print("\nCombining all meters...")
df = pd.concat(data).reset_index()

# FIXED: Rename columns (keeping P_actual)
df = df.rename(columns={
    'ReactiveEnergy': 'ReactiveCumulativeEnergyImport(+R)',
    'ActiveEnergy': 'DemandRegisterActiveEnergyCombinedTotal',
    'P_reported': 'LastAverageValueOfImportActivePower',
    'P_actual': 'ActualPowerConsumption',  # CRITICAL: Keep this for detection
    'PF': 'LastAverageValueOfPFTotal',
    'I': 'LastAverageValueOfCurrent',
    'V': 'LastAverageValueOfVoltageL',
    'S': 'ApparentPower'
})

# Add negative reactive energy
df['ReactiveCumulativeEnergyImport(-R)'] = 0

# FIXED: Final column ordering (including P_actual)
cols = [
    'MeterID', 'timestamp', 'Phase', 'CustomerType', 'Region', 'NTL_Type',
    'ActualPowerConsumption',  # ADDED
    'LastAverageValueOfImportActivePower',
    'DemandRegisterActiveEnergyCombinedTotal',
    'ReactiveCumulativeEnergyImport(+R)',
    'ReactiveCumulativeEnergyImport(-R)',
    'LastAverageValueOfPFTotal',
    'LastAverageValueOfCurrent',
    'LastAverageValueOfVoltageL',
    'ApparentPower'
]
df = df[cols]

# Summary statistics
print("\n" + "="*70)
print("DATA GENERATION COMPLETE")
print("="*70)
print(f"\nDataset Summary:")
print(f"  Total Records: {len(df):,}")
print(f"  Unique Meters: {df['MeterID'].nunique()}")
print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Normal Meters: {(df['NTL_Type'] == 'normal').sum() // HOURS}")
print(f"  NTL Meters: {(df['NTL_Type'] != 'normal').sum() // HOURS}")
print(f"\nNTL Type Distribution:")
print(df.groupby('NTL_Type')['MeterID'].nunique())
print(f"\nCustomer Type Distribution:")
print(df.groupby('CustomerType')['MeterID'].nunique())

# Save to CSV
print(f"\nSaving to: {full_path}")
df.to_csv(full_path, index=False, header=True)
print("✓ File saved successfully!")

# Data quality checks
print("\n" + "="*70)
print("DATA QUALITY CHECKS")
print("="*70)
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Power range: {df['ActualPowerConsumption'].min():.3f} - {df['ActualPowerConsumption'].max():.3f} kW")
print(f"Voltage range: {df['LastAverageValueOfVoltageL'].min():.1f} - {df['LastAverageValueOfVoltageL'].max():.1f} V")
print(f"Current range: {df['LastAverageValueOfCurrent'].min():.3f} - {df['LastAverageValueOfCurrent'].max():.3f} A")

# Calculate theft indicators
df['PowerLoss_Percent'] = 100 * (df['ActualPowerConsumption'] - df['LastAverageValueOfImportActivePower']) / (df['ActualPowerConsumption'] + 1e-6)
print(f"\nTheft Detection Preview:")
print(f"  Average power loss (NTL meters): {df[df['NTL_Type'] != 'normal']['PowerLoss_Percent'].mean():.2f}%")
print(f"  Average power loss (Normal meters): {df[df['NTL_Type'] == 'normal']['PowerLoss_Percent'].mean():.2f}%")

print("\n✓ All done! Dataset ready for NTL detection models.")
