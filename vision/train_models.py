import pickle

from proglearn.forest import UncertaintyForest, LifelongClassificationForest
from utils import load_embedded_data

n_estimators = 100
n_train = 4500  # Leave the last 500 points for validation Transfer Efficiency.

X_train, y_train, X_test, y_test = load_embedded_data(split_train=True)
lf = LifelongClassificationForest(n_estimators=n_estimators)

for t in range(10):
    uf = UncertaintyForest(n_estimators=n_estimators)

    train_x = X_train[t][0:n_train]
    train_y = y_train[t][0:n_train]

    uf.fit(train_x, train_y)
    lf.add_task(train_x, train_y, task_id=t)
    pickle.dump(uf, open("output/uf_task_%d.p" % t, "wb"))


pickle.dump(lf, open("output/lf_task_10.p", "wb"))
