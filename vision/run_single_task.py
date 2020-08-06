import pickle

from proglearn.forest import UncertaintyForest
from utils import load_embedded_data

n_estimators = 300

X_train, y_train, X_test, y_test = load_embedded_data(split_train=True, split_test=True)

uf_probs = []
for t in range(10):
    uf = UncertaintyForest(n_estimators=n_estimators)
    uf.fit(X_train[t], y_train[t])
    uf_probs.append(uf.predict_proba(X_test[t]))

pickle.dump(uf_probs, open("output/uf_probs.p", "wb"))
