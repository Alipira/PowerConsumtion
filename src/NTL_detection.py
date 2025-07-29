from typing import List


class NtlDetection:
    def __init__(self, df: pd.DataFrame, method: List=[str]):
        self._method = method
        self.df = df


    # Bagging Algorithm
    def ntl_supervised(self, labels: pd.Series, method=None):
        features = self.df.drop(columns=['MeterID','timestamp', 'cluster','NTL_Type'])
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        return clf, X_test, y_test


    # Unsupervised
    def ntl_unsupervised(self, ):
        features = self.df.drop(columns=['MeterID','timestamp', 'cluster','NTL_Type'])
        iso = IsolationForest(contamination=0.01, random_state=0)
        scores = iso.fit_predict(features)
        return scores


    def deep(self, ):
        pass


    def run(self, labels: pd.Series = None):
        if self._method == 'supervised':
            return self.ntl_supervised(labels)
        elif self._method == 'unsupervised':
            return self.ntl_unsupervised()
        else:
            raise ValueError("Unknown method. Use 'supervised' or 'unsupervised'.")
