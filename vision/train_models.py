import pickle

from proglearn.forest import UncertaintyForest, LifelongClassificationForest
from utils import load_embedded_data

n_estimators = 100

X_train, y_train, X_test, y_test = load_embedded_data(split_train=True)

lf = LifelongClassificationForest(n_estimators=n_estimators)

for t in range(10):
    uf = UncertaintyForest(n_estimators=n_estimators)
    uf.fit(X_train[t], y_train[t])
    lf.add_task(X_train[t], y_train[t], task_id=t)
    pickle.dump(uf, open("output/uf_task_%d.p" % t, "wb"))

pickle.dump(lf, open("output/lf_task_10.p", "wb"))
