def plot_daily_weekly_cycles(df: pd.DataFrame, customer_id: int):
    cust = df[df['MeterID'] == customer_id].set_index('timestamp')
    # daily cycle
    daily = cust['LastAverageValueOfImportActivePower'].groupby(cust.index.time).mean()
    plt.figure()
    plt.plot(daily.index.astype(str), daily.values)
    plt.xticks(rotation=90)
    plt.title(f"Daily cycle for customer {customer_id}")
    plt.xlabel('Time of day')
    plt.ylabel('Average Power')
    plt.show()
    # weekly cycle
    weekly = cust['LastAverageValueOfImportActivePower'].groupby(cust.index.dayofweek).mean()
    plt.figure()
    plt.bar(weekly.index, weekly.values)
    plt.title(f"Weekly cycle for customer {customer_id}")
    plt.xlabel('Day of week')
    plt.ylabel('Average Power')
    plt.show()


def compute_entropy(ts: pd.Series, bins: int = 50) -> float:
    counts, _ = np.histogram(ts, bins=bins, density=False)
    probs = counts / counts.sum()
    return entropy(probs, base=2)


def plot_entropy_all(df: pd.DataFrame):
    ent = df.groupby('MeterID')['LastAverageValueOfImportActivePower'].apply(compute_entropy)
    ent.plot(kind='hist', bins=20)
    plt.title('Distribution of Entropy across customers')
    plt.xlabel('Entropy')
    plt.show()


def cluster_load_shapes(df: pd.DataFrame, n_clusters: int = 2):
    # extract daily load shapes
    pivot = df.pivot_table(values='LastAverageValueOfImportActivePower', index='MeterID', columns=df['timestamp'].dt.hour * 4 + df['timestamp'].dt.minute // 15)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pivot.fillna(0))
    return kmeans.labels_
