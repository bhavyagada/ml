import numpy as np
from datasets import load_dataset

# all columns => ["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
# feature columns => ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
# target column => ["Species"]

#### DATA PREPARATION
dataset = load_dataset("scikit-learn/iris", split="train")
features, target = dataset.column_names[3:-1], dataset.column_names[-1]

# encode: Iris-setosa => 0, Iris-versicolor => 1, Iris-virginica => 2
dataset = dataset.class_encode_column(target)

# train set will be 80% and test set will be 20%
# seed is for reproducability of the split
split = dataset.train_test_split(train_size=0.8, test_size=0.2, seed=42)
train_ds, test_ds = split["train"].with_format('numpy'), split["test"].with_format('numpy')

# X are features, Y is target
X_train = np.stack([train_ds[feature] for feature in features], axis=1)
Y_train = train_ds[target]
X_test = np.stack([test_ds[feature] for feature in features], axis=1)
Y_test = test_ds[target]

#### GAUSSIAN NAIVE BAYES ALGORITHM
class GaussianNB:
  def fit(self, X, Y):
    self.classes = np.unique(Y) # 0, 1, 2
    self.stats = { c: { 'prior': np.mean(Y == c), 'mean': X[Y == c].mean(axis=0), 'var': X[Y == c].var(axis=0) } for c in self.classes }

  def predict(self, X):
    log_probs = np.empty((X.shape[0], len(self.classes)))
    for idx, c in enumerate(self.classes):
      stats = self.stats[c]
      log_prior = np.log(stats['prior'])
      log_likelihood = -0.5 * (np.log(2 * np.pi * stats['var']) + ((X - stats['mean'])**2 / stats['var'])).sum(axis=1)
      log_probs[:, idx] = log_prior + log_likelihood
    return self.classes[np.argmax(log_probs, axis=1)]

#### TRAINING AND PREDICTION
model = GaussianNB()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
accuracy = (y_pred == Y_test).mean() * 100
print(f"Accuracy => {accuracy:.2f}%")

#### CONFUSION MATRIX
def confusion_matrix(true, pred):
  K = np.unique(true).size
  result = np.zeros((K, K))
  for i in range(len(true)): result[true[i]][pred[i]] += 1
  return result
print(confusion_matrix(Y_test, y_pred))

#### SKLEARN FOR COMPARISON
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
X = iris.data
Y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"sklearn Accuracy => {accuracy_score(y_test, y_pred) * 100:.2f}%")

