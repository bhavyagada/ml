import numpy as np
from datasets import load_dataset

dataset = load_dataset("julien-c/titanic-survival", split="train")
features, target = ["Pclass", "Sex", "Age"], dataset.column_names[0]
dataset = dataset.class_encode_column("Sex")
dataset = dataset.class_encode_column(target)

split = dataset.train_test_split(train_size=0.8, test_size=0.2, seed=42)
train_ds, test_ds = split["train"].with_format("numpy"), split["test"].with_format("numpy")

X_train = np.stack([train_ds[feature] for feature in features], axis=1)
Y_train = train_ds[target]
X_test = np.stack([test_ds[feature] for feature in features], axis=1)
Y_test = test_ds[target]

class GaussianNB:
  def fit(self, X, Y):
    self.classes = np.unique(Y)
    self.stats = { c: { 'prior': np.mean(Y == c), 'mean': X[Y == c].mean(axis=0), 'var': X[Y == c].var(axis=0) } for c in self.classes }

  def predict(self, X):
    log_probs = np.empty((X.shape[0], len(self.classes)))
    for idx, c in enumerate(self.classes):
      stats = self.stats[c]
      log_prior = np.log(stats['prior'])
      log_likelihood = -0.5 * (np.log(2 * np.pi * stats['var']) + ((X - stats['mean'])**2 / stats['var'])).sum(axis=1)
      log_probs[:, idx] = log_prior + log_likelihood
    return self.classes[np.argmax(log_probs, axis=1)]

model = GaussianNB()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
accuracy = (y_pred == Y_test).mean() * 100
print(f"Accuracy => {accuracy:.2f}%")
