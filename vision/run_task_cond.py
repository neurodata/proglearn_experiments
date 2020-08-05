import numpy as np
import pickle

from proglearn.forest import LifelongClassificationForest

from utils import load_embedded_data

n_estimators = 300

X_train, y_train, X_test, y_test = load_embedded_data(split_train=True)

lf = LifelongClassificationForest(n_estimators=n_estimators)

for t in range(10):
    lf.add_task(X_train[t], y_train[t], task_id=t)

for t in range(10):
    # p(y | x, t)
    task_cond_probs_t = lf.predict_proba(X_test, t)

    if t == 0:
        task_cond_probs = task_cond_probs_t
    else:
        task_cond_probs = np.concatenate((task_cond_probs, task_cond_probs_t), axis=1)

pickle.dump(task_cond_probs, open("output/task_cond_probs.p", "wb"))

