import pandas as pd
import numpy as np
import os

from datetime import datetime, timedelta
from dotenv import load_dotenv


# Load .env file
load_dotenv()
csv_path = os.getenv('csv_path')

if csv_path is None:
    raise ValueError(f"Environment variable {csv_path} is not set !!!")

full_path = os.path.join(csv_path, 'ntldata.csv')

# Parameters
num_customers = 10
num_days = 10
start_date = datetime(2025, 6, 1)

data = []

# Assign first 2 customers as NTL
ntl_customers = set(range(1, 3))

for cust_id in range(1, num_customers + 1):
    cum_active = 0.0
    cum_reactive_plus = 0.0
    cum_reactive_minus = 0.0
    for day in range(num_days):
        date = start_date + timedelta(days=day)
        # Normal vs NTL behavior
        if cust_id in ntl_customers and 3 <= day <= 6:
            # NTL period
            active_inc = np.random.uniform(0, 0.5)  # near-flat consumption
            reactive_inc = -np.random.uniform(0, 0.5)  # negative reactive
            pf = np.random.uniform(0.98, 1.0)  # unnaturally high PF
            voltage = np.random.normal(240, 1)  # slight boosted voltage
        else:
            # Normal consumption
            active_inc = np.random.uniform(8, 15)
            reactive_inc = active_inc * np.random.uniform(0.4, 0.6)
            pf = np.random.uniform(0.75, 0.95)
            voltage = np.random.normal(230, 3)
        cum_active += active_inc
        cum_reactive_plus += max(reactive_inc, 0)
        cum_reactive_minus += abs(min(reactive_inc, 0))
        current = active_inc / (voltage + 1e-6) * 1000 / (np.sqrt(3))  # approximate current in A

        data.append({
            'CustomerID': cust_id,
            'Date': date.date(),
            'ReactiveCumulativeEnergyImport(+R)': round(cum_reactive_plus, 2),
            'ReactiveCumulativeEnergyImport(-R)': round(cum_reactive_minus, 2),
            'DemandRegisterActiveEnergyCombinedTotal': round(cum_active, 2),
            'LastAverageValueOfImportActivePower': round(active_inc, 2),
            'LastAverageValueOfPFTotal': round(pf, 3),
            'LastAverageValueOfCurrent': round(current, 2),
            'LastAverageValueOfVoltageL': round(voltage, 2),
            'NTL_Flag': int(cust_id in ntl_customers and 3 <= day <= 6)
        })

df = pd.DataFrame(data)

# Save CSV
df.to_csv(full_path, index=False)
