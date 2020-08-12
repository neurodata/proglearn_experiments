import pickle

from proglearn.forest import LifelongClassificationForest

n_estimators = 100

lf = LifelongClassificationForest(n_estimators=n_estimators)

tasks = [
    {"task_id": 0, "size": 500},
    {"task_id": 1, "size": 1000},
    {"task_id": 2, "size": 2000},
]

for task in tasks:

    X_train = pickle.load(open("output/X_%d_train.p" % task['size'], "rb"))
    y_train = pickle.load(open("output/y_%d_train.p" % task['size'], "rb"))

    print("Adding task %d" % task["task_id"])
    lf.add_task(X_train, y_train, task["task_id"])

pickle.dump(lf, open("output/lf_icon_trained.p", "wb"))
