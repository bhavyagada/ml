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
  def __init__(self, alpha=1e-5):
    self.smoothing = alpha

  def fit(self, X, Y):
    self.classes = np.unique(Y) # 0, 1, 2
    self.stats = {}

    for c in self.classes:
      X_c = X[Y == c]
      self.stats[c] = {
          'prior': len(X_c) / len(X),
          'mean': X_c.mean(axis=0),
          'var': X_c.var(axis=0, ddof=0) + self.smoothing # epsilon to prevent divide by zero
      }

  def _pdf(self, x, mean, var):
    return np.exp(-(x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var) # NOTE: var = sigma^2

  def predict(self, X):
    posteriors = []
    for x in X:
      probs = []
      for c in self.classes:
        prior = np.log(self.stats[c]['prior'])
        likelihood = np.sum(np.log(self._pdf(x, self.stats[c]['mean'], self.stats[c]['var'])))
        probs.append(prior + likelihood)
      posteriors.append(self.classes[np.argmax(probs)])
    return np.array(posteriors)

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

