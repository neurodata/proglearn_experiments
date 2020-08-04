import numpy as np
import pickle

from utils import load_toxic_comment, get_source_and_target

from proglearn.forest import LifelongClassificationForest


def fit_source_tasks(source_tasks, multitask_id, n_estimators=10, verbose=True):

    np.random.seed(12345)

    l2f = LifelongClassificationForest(n_estimators=n_estimators)
    # Fit source tasks.
    for task in source_tasks:
        print("----------------------------------------")
        print("TASK:", task["name"])
        print("----------------------------------------")

        # Load the train and test data.
        X_train, y_train, X_test, y_test = task["load"](
            verbose=verbose, subsample_frac=task["subsample_frac"]
        )

        l2f.add_task(X_train, y_train, task_id=task["id"])

    pickle.dump(
        l2f,
        open("output/l2f_source_trained_%d_%d.p" % (n_estimators, multitask_id), "wb"),
    )


def fit_task_priors(X_train_pooled, t_train_pooled, multitask_id, n_estimators=10):
    l2f = LifelongClassificationForest(n_estimators=n_estimators)
    l2f.add_task(X_train_pooled, t_train_pooled, task_id=0)
    pickle.dump(
        l2f,
        open(
            "output/l2f_task_prior_trained_%d_%d.p" % (n_estimators, multitask_id), "wb"
        ),
    )


def predict_task_priors(
    X_test_pooled,
    t_test_pooled,
    target_task,
    multitask_id,
    verbose=True,
    n_estimators=10,
):
    l2f = pickle.load(
        open(
            "output/l2f_task_prior_trained_%d_%d.p" % (n_estimators, multitask_id), "rb"
        )
    )

    # Just to see if it can discriminate among the source tasks.
    task_prior_probs_source = l2f.predict_proba(X_test_pooled, 0)
    pickle.dump(
        task_prior_probs_source,
        open(
            "output/task_prior_probs_source_%d_%d.p" % (n_estimators, multitask_id),
            "wb",
        ),
    )
    pickle.dump(
        t_test_pooled, open("output/source_task_labels%d.p" % multitask_id, "wb"),
    )

    # These are the task priors for the toxic comment set.
    np.random.seed(12345)
    X_train, y_train, X_test, y_test = target_task["load"](verbose=verbose)
    task_prior_probs_target = l2f.predict_proba(X_test, 0)

    pickle.dump(
        task_prior_probs_target,
        open(
            "output/task_prior_probs_target_%d_%d.p" % (n_estimators, multitask_id),
            "wb",
        ),
    )
    pickle.dump(
        y_test, open("output/target_y_test_%d.p" % multitask_id, "wb"),
    )


def compute_posteriors(
    source_tasks, target_task, multitask_id, n_estimators=10, verbose=True
):

    # Load toxic comment data.
    X_train, y_train, X_test, y_test = target_task["load"](verbose=verbose)

    l2f = pickle.load(
        open("output/l2f_source_trained_%d_%d.p" % (n_estimators, multitask_id), "rb")
    )

    # which task's posterior predictor.
    for t, task in enumerate(source_tasks):
        probs_t = l2f.predict_proba(X_test, task["id"])
        if t == 0:
            probs = probs_t
        else:
            probs = np.concatenate((probs, probs_t), axis=1)

    # Save test data again just to make sure posteriors match.
    pickle.dump(y_test, open("output/y_test_%d.p" % multitask_id, "wb"))
    pickle.dump(probs, open("output/probs_%d.p" % multitask_id, "wb"))


def load_pooled_data(source_tasks, verbose=True):

    np.random.seed(123)
    for task in source_tasks:
        X_train, _, X_test, _ = task["load"](
            verbose=verbose, subsample_frac=task["subsample_frac"]
        )

        # Task labels.
        t_train = task["id"] * np.ones(len(X_train))
        t_test = task["id"] * np.ones(len(X_test))

        if task["id"] == 0:
            X_train_pooled = X_train
            t_train_pooled = t_train
            X_test_pooled = X_test
            t_test_pooled = t_test
        else:
            X_train_pooled = np.concatenate((X_train_pooled, X_train), axis=0)
            t_train_pooled = np.concatenate((t_train_pooled, t_train), axis=0)
            X_test_pooled = np.concatenate((X_test_pooled, X_test), axis=0)
            t_test_pooled = np.concatenate((t_test_pooled, t_test), axis=0)

    return X_train_pooled, t_train_pooled, X_test_pooled, t_test_pooled


def run_zero_shot(source_names, target_name, multitask_id, n_estimators, verbose):
    source_tasks, target_task = get_source_and_target(source_names, target_name)

    # Fit task priors p(t | x).
    X_train_pooled, t_train_pooled, X_test_pooled, t_test_pooled = load_pooled_data(
        source_tasks
    )
    fit_task_priors(
        X_train_pooled, t_train_pooled, multitask_id, n_estimators=n_estimators
    )
    predict_task_priors(
        X_test_pooled,
        t_test_pooled,
        target_task,
        multitask_id,
        n_estimators=n_estimators,
    )

    # Fit classifiers p(y, | x, t)
    fit_source_tasks(
        source_tasks, multitask_id, n_estimators=n_estimators, verbose=verbose
    )
    compute_posteriors(
        source_tasks,
        target_task,
        multitask_id,
        n_estimators=n_estimators,
        verbose=verbose,
    )
