import pickle

from utils import get_source_and_target, load_toxic_comment
from proglearn.forest import LifelongClassificationForest, UncertaintyForest
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
lf = LifelongClassificationForest(n_estimators=n_estimators)
for task in source_tasks:
    X_train, y_train, _, _ = task["load"](
        verbose=verbose, subsample_frac=task["subsample_frac"]
    )
    lf.add_source_task(X_train, y_train, task_id=task["id"])

# Save the transfer forest with these source task.
pickle.dump(lf, open("output/lf_source_trained_task_aware_%d.p" % n_estimators, "wb"))

# Load target data, train voters, and compute accuracy.
X_train_full, y_train_full, X_test, y_test = load_toxic_comment(verbose=verbose)

lf_accs = []
uf_accs = []
for i, subsample_frac in enumerate(subsample_fracs):

    tf_accs_n = []
    uf_accs_n = []
    for s in range(n_sims):
        _, X_train, _, y_train = train_test_split(
            X_train_full, y_train_full, test_size=subsample_frac,
        )

        lf = pickle.load(
            open("output/lf_source_trained_task_aware_%d.p" % n_estimators, "rb")
        )
        lf.add_target_task(X_train, y_train, task_id=target_task["id"])
        y_pred = lf.predict(X_test)
        tf_accs_n.append(
            classification_report(y_test, y_pred, zero_division=1, output_dict=True)
        )

        uf = UncertaintyForest(n_estimators=(len(source_tasks) + 1) * n_estimators)
        uf.fit(X_train, y_train)
        y_pred = uf.predict(X_test)
        uf_accs_n.append(
            classification_report(y_test, y_pred, zero_division=1, output_dict=True)
        )

    lf_accs.append(tf_accs_n)
    uf_accs.append(uf_accs_n)

pickle.dump(subsample_fracs, open("output/lf_subsample_fracs.p", "wb"))
pickle.dump(lf_accs, open("output/lf_accs_task_aware.p", "wb"))
pickle.dump(uf_accs, open("output/uf_accs_task_aware.p", "wb"))

