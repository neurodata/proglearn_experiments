import numpy as np
import pickle

from proglearn.forest import LifelongClassificationForest
from sklearn.metrics import accuracy_score

from utils import load_embedded_data

n_estimators = 100

X_train, t_train, X_test, t_test = load_embedded_data(task_prior=True)

train_idx = pickle.load(open("output/train_idx_task_prior_small.p", "rb"))

X_train = X_train[train_idx]
t_train = t_train[train_idx]

lf = LifelongClassificationForest(n_estimators=n_estimators)
lf.add_task(X_train, t_train, task_id=0)

task_priors = lf.predict_proba(X_test, 0)
pickle.dump(task_priors, open("output/task_priors_small.p", "wb"))

t_pred = np.argmax(task_priors, axis=1)
acc = accuracy_score(t_test, t_pred)

print("Accuracy of ResNet50 + L2F (small) on task prediction:", acc)

