import numpy as np
import pandas as pd


class PowerConsumptionSimulator:
    def __init__(self, n_customers: int, n_days: int = 365):
        self.n_customers = n_customers
        self.n_days = n_days
        self._hours_per_day = 24

        # Variability factor to simulate different customer consumption patterns during the day/night and seasons
        # hourly: 24 multipliers for hours 0..23 (midnight..23:00)
        # seasonal: multipliers for typical seasons
        # weekday_weekend: scalar to adjust overall daily usage on weekends vs weekdays
        self.variability = {
            'residential': {
                'hourly': [
                    0.60,  # 00:00
                    0.55,  # 01:00
                    0.50,  # 02:00
                    0.50,  # 03:00
                    0.55,  # 04:00
                    0.70,  # 05:00 - wakeup begins
                    1.00,  # 06:00
                    1.20,  # 07:00 - morning peak
                    1.10,  # 08:00
                    0.90,  # 09:00
                    0.85,  # 10:00
                    0.85,  # 11:00
                    0.90,  # 12:00
                    0.90,  # 13:00
                    0.85,  # 14:00
                    0.90,  # 15:00
                    1.00,  # 16:00
                    1.30,  # 17:00 - evening ramp
                    1.50,  # 18:00 - evening peak
                    1.60,  # 19:00 - peak usage
                    1.50,  # 20:00
                    1.30,  # 21:00
                    1.00,  # 22:00
                    0.80   # 23:00
                ],
                'seasonal': {
                    'winter': 1.15,  # heating lighting increases
                    'spring': 0.95,
                    'summer': 1.25,  # air conditioning increases in many places
                    'autumn': 1.00
                },
                'weekday_weekend': {
                    'weekday': 1.00,
                    'weekend': 1.10  # people at home more on weekends
                }
            },

            'commercial': {
                'hourly': [
                    0.20,  # 00
                    0.18,  # 01
                    0.15,  # 02
                    0.15,  # 03
                    0.18,  # 04
                    0.30,  # 05
                    0.40,  # 06 - early prep
                    0.80,  # 07
                    1.00,  # 08 - business hours start
                    1.10,  # 09
                    1.05,  # 10
                    1.00,  # 11
                    0.95,  # 12 - lunch dip
                    0.95,  # 13
                    1.00,  # 14
                    1.05,  # 15
                    1.00,  # 16
                    0.80,  # 17 - close/cleaning
                    0.50,  # 18
                    0.30,  # 19
                    0.25,  # 20
                    0.22,  # 21
                    0.20,  # 22
                    0.20   # 23
                ],
                'seasonal': {
                    'winter': 1.05,
                    'spring': 0.95,
                    'summer': 1.10,  # AC in offices/shop increases
                    'autumn': 1.00
                },
                'weekday_weekend': {
                    'weekday': 1.00,
                    'weekend': 0.70  # many commercial entities reduce operations on weekends
                }
            },

            'industrial': {
                'hourly': [
                    0.85,  # 00 - overnight lower staffing
                    0.85,  # 01
                    0.85,  # 02
                    0.85,  # 03
                    0.85,  # 04
                    0.90,  # 05
                    0.95,  # 06 - shift startup
                    1.00,  # 07
                    1.05,  # 08 - full production
                    1.05,  # 09
                    1.05,  # 10
                    1.05,  # 11
                    1.05,  # 12
                    1.05,  # 13
                    1.05,  # 14
                    1.05,  # 15
                    1.05,  # 16
                    1.00,  # 17 - small ramp down
                    0.98,  # 18
                    0.95,  # 19
                    0.92,  # 20
                    0.90,  # 21
                    0.90,  # 22
                    0.88   # 23
                ],
                'seasonal': {
                    'winter': 1.02,
                    'spring': 0.98,
                    'summer': 1.03,
                    'autumn': 0.97
                },
                'weekday_weekend': {
                    'weekday': 1.00,
                    'weekend': 0.95  # many industrial sites reduce operations slightly on weekends
                }
            }
        }

        # KwH consumption ranges for different customer types (mean , std)
        self.cusotmer_types_consumption = {
            'residential': (5, 15),
            'commercial': (50, 20),
            'industrial': (200, 50)
        }

    def generate_normal_consumption(self, customer_type: str | None = 'residential') -> pd.DataFrame:
        """Generates normal power consumption data for all customers over the specified number of days."""
        data = {}
        customer_ids = [f'customer_{i+1}' for i in range(self.n_customers)]
        customer_types = np.random.choice(list(self.cusotmer_types_consumption.keys()), size=self.n_customers, p=[0.6, 0.25, 0.15])
        #FIXME:
        # if customer_type is None:
        #     customer_types = np.random.choice(list(self.cusotmer_types_consumption.keys()), size=self.n_customers, p=[0.6, 0.25, 0.15])

        for cust_id, cust_type in zip(customer_ids, customer_types):
            daily_consumption = []
            for day in range(self.n_days):
                day_of_week = day % 7
                season = (day // 90) % 4  # 0:winter, 1:spring, 2:summer, 3:autumn
                if season == 0:
                    season = 'winter'
                elif season == 1:
                    season = 'spring'
                elif season == 2:
                    season = 'summer'
                else:
                    season = 'autumn'
                weekday_weekend = 'weekend' if day_of_week in [5, 6] else 'weekday'

                for hour in range(self._hours_per_day):
                    hourly_var = self.variability[cust_type]['hourly'][hour]
                    seasonal_var = self.variability[cust_type]['seasonal'][season]
                    ww_var = self.variability[cust_type]['weekday_weekend'][weekday_weekend]

                    mean_consumption, std_consumption = self.cusotmer_types_consumption[cust_type]
                    consumption = np.random.normal(mean_consumption, std_consumption)

                    noise = np.random.normal(0, 0.05 * consumption)  # 5% noise
                    adjusted_consumption = consumption * hourly_var * seasonal_var * ww_var
                    daily_consumption.append(max(adjusted_consumption + noise, 0))  # Ensure no negative consumption

            data[cust_id] = daily_consumption

        index = pd.date_range(start='2023-01-01', periods=self.n_days * self._hours_per_day, freq='h')
        df = pd.DataFrame(data, index=index)
        return df

    def inject_ntl_patterns(self, df: pd.DataFrame, ntl_ratio: float = 0.15, reduction_ratio: float = 0.10) -> pd.DataFrame:
        """Injects Non-Technical Loss (NTL) patterns into the power consumption data."""
        n_ntl_customers = int(len(df['customer_ids'].unique()) * ntl_ratio)
        ntl_customers = np.random.choice(
            df['customer_ids'].unique(),
            size=n_ntl_customers,
            replace=False
        )
        df_ntl = df.copy(deep=True)

        # Inject theft patterns
        for customer in ntl_customers:
            mask = df_ntl['customer_ids'] == customer
            ntl_types = np.random.choice(['theft', 'meter_tampering', 'irregular'])

            if ntl_types == 'theft':
                # Sudden drop in consumption (direct theft)
                for day in range(self.n_days):
                    if np.random.rand() < 0.3:  # 30% chance of theft on any given day
                        start_hour = np.random.randint(0, self._hours_per_day - 4)
                        df_ntl.loc[df.index[day * self._hours_per_day + start_hour : day * self._hours_per_day + start_hour + 4], customer] *= 0.5  # 50% reduction
            elif ntl_types == 'meter_tampering':
                # Gradual reduction in reported consumption (meter tampering)
                for day in range(self.n_days):
                    if np.random.rand() < 0.2:  # 20% chance of tampering on any given day
                        start_hour = np.random.randint(0, self._hours_per_day - 6)
                        df_ntl.loc[df.index[day * self._hours_per_day + start_hour : day * self._hours_per_day + start_hour + 6], customer] *= 0.7  # 30% reduction
            else:
                # Irregular consumption patterns
                for day in range(self.n_days):
                    if np.random.rand() < 0.25:  # 25% chance of irregularity on any given day
                        start_hour = np.random.randint(0, self._hours_per_day - 5)
                        fluctuation = np.random.choice([0.5, 1.5])  # Either spike or drop
                        df_ntl.loc[df.index[day * self._hours_per_day + start_hour : day * self._hours_per_day + start_hour + 5], customer] *= fluctuation
            return df_ntl

    # def inject_ntl_patterns(self, df: pd.DataFrame, theft_fraction: float = 0.15, reduction_fraction: float = 0.10):
    #     """Injects Non-Technical Loss (NTL) patterns into the power consumption data."""
    #     n_theft_customers = int(len(df['customer_id'].unique()) * theft_fraction)
    #     n_reduction_customers = int(self.n_customers * reduction_fraction)

    #     theft_customers = np.random.choice(df.columns, size=n_theft_customers, replace=False)
    #     remaining_customers = list(set(df.columns) - set(theft_customers))
    #     reduction_customers = np.random.choice(remaining_customers, size=n_reduction_customers, replace=False)

    #     # Inject theft patterns
    #     for cust in theft_customers:
    #         for day in range(self.n_days):
    #             if np.random.rand() < 0.3:  # 30% chance of theft on any given day
    #                 start_hour = np.random.randint(0, self._hours_per_day - 4)
    #                 df.loc[df.index[day * self._hours_per_day + start_hour: day * self._hours_per_day + start_hour + 4], cust] *= 0.5  # 50% reduction

    #     # Inject reduction patterns
    #     for cust in reduction_customers:
    #         for day in range(self.n_days):
    #             if np.random.rand() < 0.2:  # 20% chance of reduction on any given day
    #                 start_hour = np.random.randint(0, self._hours_per_day - 6)
    #                 df.loc[df.index[day * self._hours_per_day + start_hour: day * self._hours_per_day + start_hour + 6], cust] *= 0.7  # 30% reduction

    #     return df
