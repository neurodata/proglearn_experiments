import numpy as np
import pickle

from utils import get_source_and_target, load_toxic_comment
from proglearn.forest import TransferForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Hyperparameters.
verbose = True
n_estimators = 10
# subsample_fracs = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
subsample_fracs = [0.00003, 0.0001]

# Load source data.
source_names = ["yelp", "imdb", "amazon"]
target_name = "toxic_comment"
source_tasks, target_task = get_source_and_target(source_names, target_name)

# Train transformers on source data.
tf = TransferForest(n_estimators=n_estimators)
for task in source_tasks:
    X_train, y_train, _, _ = task['load'](
        verbose=verbose, 
        subsample_frac=task['subsample_frac']
    )
    tf.add_source_task(X_train, y_train, task_id=task['id'])

# Save the transfer forest with these source task.
pickle.dump(tf, open("output/tf_source_trained_%d.p" % n_estimators, "wb"))

# Load target data, train voters, and compute accuracy.
X_train_full, y_train_full, X_test, y_test = load_toxic_comment(verbose=verbose)

np.random.seed(12345)
accs = []
for subsample_frac in subsample_fracs:
    _, X_train, _, y_train = train_test_split(
        X_train_full, 
        y_train_full, 
        test_size=subsample_frac, 
        random_state=42
    )

    tf = pickle.load(open("output/tf_source_trained_%d.p" % n_estimators, "rb"))
    tf.add_target_task(X_train, y_train, task_id=target_task['id'])
    y_pred = tf.predict(X_test)
    accs.append(accuracy_score(y_pred, y_test))

pickle.dump(subsample_fracs, open("output/tf_subsample_fracs.p", "wb"))
pickle.dump(accs, open("output/tf_accs.p", "wb"))







