import numpy as np
import pickle

import keras
import keras.layers as layers

from proglearn.network import LifelongClassificationNetwork


def pull_data(num_points_per_task=500, num_tasks=10, shift=1):
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    data_x = np.concatenate([X_train, X_test])
    data_y = np.concatenate([y_train, y_test])

    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    batch_per_task = 5000 // num_points_per_task
    sample_per_class = num_points_per_task // num_tasks
    test_data_slot = 100 // batch_per_task

    for task in range(num_tasks):
        for batch in range(batch_per_task):
            for class_no in range(task * 10, (task + 1) * 10, 1):
                indx = np.roll(idx[class_no], (shift - 1) * 100)

                if batch == 0 and class_no == 0 and task == 0:
                    train_x = x[
                        indx[batch * sample_per_class : (batch + 1) * sample_per_class],
                        :,
                    ]
                    train_y = y[
                        indx[batch * sample_per_class : (batch + 1) * sample_per_class]
                    ]
                    test_x = x[
                        indx[
                            batch * test_data_slot
                            + 500 : (batch + 1) * test_data_slot
                            + 500
                        ],
                        :,
                    ]
                    test_y = y[
                        indx[
                            batch * test_data_slot
                            + 500 : (batch + 1) * test_data_slot
                            + 500
                        ]
                    ]
                else:
                    train_x = np.concatenate(
                        (
                            train_x,
                            x[
                                indx[
                                    batch
                                    * sample_per_class : (batch + 1)
                                    * sample_per_class
                                ],
                                :,
                            ],
                        ),
                        axis=0,
                    )
                    train_y = np.concatenate(
                        (
                            train_y,
                            y[
                                indx[
                                    batch
                                    * sample_per_class : (batch + 1)
                                    * sample_per_class
                                ]
                            ],
                        ),
                        axis=0,
                    )
                    test_x = np.concatenate(
                        (
                            test_x,
                            x[
                                indx[
                                    batch * test_data_slot
                                    + 500 : (batch + 1) * test_data_slot
                                    + 500
                                ],
                                :,
                            ],
                        ),
                        axis=0,
                    )
                    test_y = np.concatenate(
                        (
                            test_y,
                            y[
                                indx[
                                    batch * test_data_slot
                                    + 500 : (batch + 1) * test_data_slot
                                    + 500
                                ]
                            ],
                        ),
                        axis=0,
                    )

    train_x = train_x.reshape(10, 5000, 32, 32, 3)
    train_y = train_y.reshape(10, 5000, 1)
    test_x = test_x.reshape(10, 1000, 32, 32, 3)
    test_y = test_y.reshape(10, 1000, 1)

    # Subsample to 500 data points per task.
    idx = np.concatenate([np.arange(t * 500, t * 500 + 50) for t in range(num_tasks)])
    train_x = train_x[:, idx, :, :, :]
    train_y = train_y[:, idx, :]

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }

    pickle.dump(data, open("experiments/cifar_exp_task_obv/data/data.p", "wb"))


def load_data():

    data = pickle.load(open("experiments/cifar_exp_task_obv/data/data.p", "rb"))
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    print("train_x shape:", train_x.shape)
    print("train_y shape:", train_y.shape)
    print("test_x shape:", test_x.shape)
    print("test_y shape:", test_y.shape)

    return train_x, train_y, test_x, test_y


def init_network(input_shape):

    network = keras.Sequential()
    network.add(
        layers.Conv2D(
            filters=16, kernel_size=(3, 3), activation="relu", input_shape=input_shape,
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=2, padding="same", activation="relu"
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=2, padding="same", activation="relu"
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=254,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )

    network.add(layers.Flatten())
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(2000, activation="relu"))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(2000, activation="relu"))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(units=10, activation="softmax"))

    return network


def fit_model(train_x, train_y, num_tasks=10):
    network = init_network(train_x.shape[1:])
    l2n = LifelongClassificationNetwork(network=network, verbose=True)

    # TODO remove 2 and put num tasks.
    for t in range(2):
        print("TRAINING TASK: ", t)
        print("-------------------------------------------------------------------")
        classes = np.unique(train_y[t])
        l2n.add_task(X=train_x[t], y=train_x[t], decider_kwargs={"classes": classes})
        print("-------------------------------------------------------------------")

    return l2n


def compute_posteriors(test_x, test_y, l2n, num_tasks=10):

    classes = np.array(l2n.task_id_to_decider[0].classes)
    probs = l2n.predict_proba(test_x, 0)

    for t in range(1, num_tasks):
        probs = np.concatenate((probs, l2n.predict_proba(test_x, t)), axis=1)
        classes = np.concatenate((classes, np.array(l2n.task_id_to_decider[t].classes)))

    # Save test data again just to make sure posteriors match.
    pickle.dump(test_x, open("experiments/cifar_exp_task_obv/data/test_x.p", "wb"))
    pickle.dump(test_y, open("experiments/cifar_exp_task_obv/data/test_y.p", "wb"))
    pickle.dump(classes, open("experiments/cifar_exp_task_obv/output/classes.p", "wb"))
    pickle.dump(probs, open("experiments/cifar_exp_task_obv/output/probs.p", "wb"))


# Run experiment.

# If first time, run pull_data to get the CIFAR 100 dataset.
# pull_data()

train_x, train_y, test_x, test_y = load_data()
l2n = fit_model(train_x, train_y)


# Check accuracy on two tasks.

# Posteriors are of the form [n_test * 100],
# where the first 10 columns is the psoterior
# from task 1, next 10 for task 2, etc.
# compute_posteriors(test_x, test_y, l2n)

