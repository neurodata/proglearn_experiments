import numpy as np
import pickle

from utils import get_source_and_target, load_toxic_comment
from proglearn.forest import TransferForest, UncertaintyForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Hyperparameters.
verbose = True
n_estimators = 10
n_sims = 10  # number of times to resample the target data.
subsample_fracs = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
# subsample_fracs = [0.0001, 0.0003]

# Load source data.
source_names = ["yelp", "imdb", "amazon"]
target_name = "toxic_comment"
source_tasks, target_task = get_source_and_target(source_names, target_name)

# Train transformers on source data.
# tf = TransferForest(n_estimators=n_estimators)
# for task in source_tasks:
#     X_train, y_train, _, _ = task['load'](
#         verbose=verbose,
#         subsample_frac=task['subsample_frac']
#     )
#     tf.add_source_task(X_train, y_train, task_id=task['id'])

# # Save the transfer forest with these source task.
# pickle.dump(tf, open("output/tf_source_trained_%d.p" % n_estimators, "wb"))

# Load target data, train voters, and compute accuracy.
X_train_full, y_train_full, X_test, y_test = load_toxic_comment(verbose=verbose)

tf_accs = []
uf_accs = []
for i, subsample_frac in enumerate(subsample_fracs):

    tf_accs_n = []
    uf_accs_n = []
    for s in range(n_sims):
        _, X_train, _, y_train = train_test_split(
            X_train_full, y_train_full, test_size=subsample_frac,
        )

        tf = pickle.load(open("output/tf_source_trained_%d.p" % n_estimators, "rb"))
        tf.add_target_task(X_train, y_train, task_id=target_task["id"])
        y_pred = tf.predict(X_test)
        tf_accs_n.append(classification_report(y_test, y_pred, zero_division=1))

        uf = UncertaintyForest(n_estimators=len(source_tasks) * n_estimators)
        uf.fit(X_train, y_train)
        y_pred = uf.predict(X_test)
        uf_accs_n.append(classification_report(y_test, y_pred, zero_division=1))

    tf_accs.append(tf_accs_n)
    uf_accs.append(uf_accs_n)

pickle.dump(subsample_fracs, open("output/tf_subsample_fracs.p", "wb"))
pickle.dump(tf_accs, open("output/tf_accs.p", "wb"))
pickle.dump(uf_accs, open("output/uf_accs.p", "wb"))

