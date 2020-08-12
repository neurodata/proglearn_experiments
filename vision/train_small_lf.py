import numpy as np
import pickle

from proglearn.forest import LifelongClassificationForest
from utils import load_embedded_data

n_estimators = 100
n_train = 500  # Leave the last 500 for validation.

X_train, y_train, X_test, y_test = load_embedded_data(split_train=True)

lf = LifelongClassificationForest(n_estimators=n_estimators)

for t in range(10):
    train_idx = np.random.choice(5000, size=n_train, replace=False)
    val_idx = np.delete(np.arange(5000), train_idx)
    pickle.dump(train_idx, open("output/train_idx_small_%d.p" % t, "wb"))
    pickle.dump(val_idx, open("output/val_idx_small_%d.p" % t, "wb"))

    train_x = X_train[t][train_idx]
    train_y = y_train[t][train_idx]

    # uf = UncertaintyForest(n_estimators=n_estimators)
    # uf.fit(train_x, train_y)
    print("Adding task %d" % t)
    lf.add_task(train_x, train_y, task_id=t)

pickle.dump(lf, open("output/lf_task_10_small.p", "wb"))
