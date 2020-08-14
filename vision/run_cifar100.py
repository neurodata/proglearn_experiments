import numpy as np
import pickle
from utils import load_embedded_data
from proglearn.forest import LifelongClassificationForest
from sklearn.metrics import accuracy_score

n_estimators = 100

X_train, y_train, X_test, y_test = load_embedded_data(split_train=False)


train_idx = pickle.load(open("output/train_idx_task_prior_small.p", "rb"))

idx = train_idx[0]
for t in range(1, 10):
    idx = np.concatenate((idx, train_idx[t]), axis=0)

X_train = X_train[idx]
y_train = y_train[idx]

lf = LifelongClassificationForest(n_estimators=n_estimators)
lf.add_task(X_train, y_train, task_id=0)

y_pred = lf.predict(X_test, 0)
acc = accuracy_score(y_test, y_pred)

print("Accuracy of ResNet50 + L2F on CIFAR 100:", acc)
pickle.dump(acc, open("output/cifar100_lf_acc_small.p", "wb"))

