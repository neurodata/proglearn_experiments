import numpy as np
import pickle

from proglearn.forest import LifelongClassificationForest
from sklearn.metrics import accuracy_score

from utils import load_embedded_data

n_estimators = 100
n_train = 5000

X_train, t_train, X_test, t_test = load_embedded_data(task_prior=True)

# train_idx = np.random.choice(5000, size=n_train, replace=False)
# val_idx = np.delete(np.arange(5000), train_idx)
# pickle.dump(train_idx, open("output/train_idx_small_%d.p" % t, "wb"))
# pickle.dump(val_idx, open("output/val_idx_small_%d.p" % t, "wb"))

lf = LifelongClassificationForest(n_estimators=n_estimators)
lf.add_task(X_train, t_train, task_id=0)

task_priors = lf.predict_proba(X_test, 0)
pickle.dump(task_priors, open("output/task_priors.p", "wb"))

t_pred = np.argmax(task_priors, axis=1)
acc = accuracy_score(t_test, t_pred)

print("Accuracy of ResNet50 + L2F on task prediction:", acc)

