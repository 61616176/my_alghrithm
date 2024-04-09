import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

class SMOTE_ENC:
    def __init__(self, t, n_percent, k, m=0):
        self.t = t
        self.n_percent = n_percent
        self.k = k
        self.m = m  # Median of standard deviation of continuous features when c > 0
        self.ir = t / s  # Imbalance ratio

    def _calculate_oversampling_factor(self, X, y, feature, label):
        e = np.sum(y == self.t)  # Total number of 'l' labelled instances
        e_hat = e * self.ir
        o = np.sum((y == self.t) & (X[:, feature] == label))
        chi = (o - e_hat) / e_hat
        if self.m > 0:
            l = chi * self.m
        else:
            l = chi
        return l

    def _apply_oversampling(self, X, y, categorical_features, continuous_features):
        new_samples = []
        for i in range(X.shape[0]):
            if y[i] == self.t:
                nearest_neighbors = self._get_nearest_neighbors(X, i, self.k)
                majority_value = self._determine_majority_value(nearest_neighbors, categorical_features)
                new_sample = X[i].copy()
                new_sample[continuous_features] = np.random.rand(len(continuous_features))
                new_sample[categorical_features] = majority_value
                new_samples.append(new_sample)
        return np.array(new_samples)

    def fit(self, X, y):
        categorical_features = np.where(X.dtypes == 'object')[0]
        continuous_features = np.where(X.dtypes == 'float64')[0]
        if continuous_features.size == 0:
            self.m = 0
        else:
            self.m = np.median(np.std(X[:, continuous_features], axis=0))

        # Calculate oversampling amount for each categorical feature
        for feature in categorical_features:
            labels = np.unique(X[:, feature])
            for label in labels:
                l = self._calculate_oversampling_factor(X, y, feature, label)
                # Apply SMOTE for each label
                synthetic_samples = self._apply_oversampling(X, y, [feature], continuous_features)
                for sample in synthetic_samples:
                    self._invert_encode(X, y, sample, feature, label, l)

        # Update the dataset with the synthetic samples
        new_X = np.vstack((X, np.array(new_samples)))
        new_y = np.hstack((y, np.full(len(new_samples), self.t)))
        return new_X, new_y

    def _get_nearest_neighbors(self, X, instance_index, k):
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X[instance_index:instance_index + 1])
        return X[indices[0][1:]]

    def _determine_majority_value(self, neighbors, categorical_features):
        majority_value = {}
        for feature in categorical_features:
            values, counts = np.unique(neighbors[:, feature], return_counts=True)
            majority_value[feature] = values[np.argmax(counts)]
        return majority_value

    def _invert_encode(self, X, y, new_sample, feature, original_label, l):
        # In this simplified version, we do not perform inverse encoding as it requires
        # the original dataset information for reverting the encoding back.
        pass

# Example usage:
# X_train - training data features
# y_train - training data labels
# t - number of minority class samples in training set
# n_percent - amount of over-sampling
# k - number of nearest neighbors to be considered
# X_train_enc, y_train_enc - encoded training data and labels with synthetic samples

# Create SMOTE-ENC instance
smote_enc = SMOTE_ENC(t, n_percent, k)

# Apply SMOTE-ENC to generate synthetic samples
X_train_enc, y_train_enc = smote_enc.fit(X_train, y_train)