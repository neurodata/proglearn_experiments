import numpy as np
import pickle

from utils import load_toxic_comment

from proglearn.forest import LifelongClassificationForest


def fit_source_tasks(source_tasks, n_estimators=10, verbose=True):

    np.random.seed(12345)

    # Fit source tasks.
    for task in source_tasks:
        print("----------------------------------------")
        print("TASK:", task["name"])
        print("----------------------------------------")

        # Load the train and test data.
        X_train, y_train, X_test, y_test = task["load"](
            verbose=verbose, subsample_frac=task["subsample_frac"]
        )

        l2f = LifelongClassificationForest(n_estimators=n_estimators)
        l2f.add_task(X_train, y_train, task_id=task["id"])

    pickle.dump(l2f, open("output/l2f_source_trained_%d.p" % n_estimators, "wb"))

    return l2f


def compute_posteriors(l2f, source_tasks, verbose=True):

    # Load toxic comment data.
    # TODO Add binarize argument.
    X_train, y_train, X_test, y_test = load_toxic_comment(verbose=verbose)

    # which task's posterior predictor.
    for t in range(len(source_tasks)):
        probs_t = l2f.predict_proba(X_test, t)
        if t == 0:
            probs = probs_t
        else:
            probs = np.concatenate((probs, probs_t), axis=1)

    # Save test data again just to make sure posteriors match.
    pickle.dump(y_test, open("output/y_test.p", "wb"))
    pickle.dump(probs, open("output/probs.p", "wb"))

