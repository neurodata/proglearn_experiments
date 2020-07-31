import numpy as np
import pickle

import keras
from keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, Dense, MaxPooling2D

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

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }

    pickle.dump(data, open("data/data.p", "wb"))


def load_data(num_tasks=10, split=True):

    data = pickle.load(open("data/data.p", "rb"))
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    if split:
        train_x = train_x.reshape(10, 5000, 32, 32, 3)
        train_y = train_y.reshape(10, 5000)
        test_x = test_x.reshape(10, 1000, 32, 32, 3)
        test_y = test_y.reshape(10, 1000)

    # Subsample to 500 data points per task.
    # train_x = train_x[:, 0:500, :, :, :]
    # train_y = train_y[:, 0:500]

    print("train_x shape:", train_x.shape)
    print("train_y shape:", train_y.shape)
    print("test_x shape:", test_x.shape)
    print("test_y shape:", test_y.shape)

    return train_x, train_y, test_x, test_y


def will_net(input_shape, num_outputs=10):

    network = keras.Sequential()
    network.add(
        Conv2D(
            filters=16, kernel_size=(3, 3), activation="relu", input_shape=input_shape,
        )
    )
    network.add(BatchNormalization())
    network.add(
        Conv2D(
            filters=32, kernel_size=(3, 3), strides=2, padding="same", activation="relu"
        )
    )
    network.add(BatchNormalization())
    network.add(
        Conv2D(
            filters=64, kernel_size=(3, 3), strides=2, padding="same", activation="relu"
        )
    )
    network.add(BatchNormalization())
    network.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(BatchNormalization())
    network.add(
        Conv2D(
            filters=254,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )

    network.add(Flatten())
    network.add(BatchNormalization())
    network.add(Dense(2000, activation="relu"))
    network.add(BatchNormalization())
    network.add(Dense(2000, activation="relu"))
    network.add(BatchNormalization())
    network.add(Dense(units=num_outputs, activation="softmax"))

    return network


def weiwei_net(input_shape, num_outputs=10):
    
    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(num_outputs, activation='softmax'))

    return model


def fit_model(train_x, train_y, num_tasks=10, lr=0.001, epochs=100, verbose=False, batch_size=32):
    # Dimensions 0 and 1 are task and sample, respectively.
    network = weiwei_net(train_x.shape[2:])
    l2n = LifelongClassificationNetwork(
        network=network,  
        epochs=epochs,
        verbose=verbose,
        lr=lr,
        batch_size=batch_size
    )

    for t in range(num_tasks):
        print("TRAINING TASK: ", t)
        print("-------------------------------------------------------------------")
        classes = np.unique(train_y[t])
        l2n.add_task(X=train_x[t], y=train_y[t], decider_kwargs={"classes": classes})
        print("-------------------------------------------------------------------")

    return l2n


def run_exp_100(train_x, train_y):
    network = weiwei_net(train_x.shape[1:])
    l2n = LifelongClassificationNetwork(network=network, lr=0.001)
    classes = np.unique(train_y)
    l2n.add_task(X=train_x, y=train_y, decider_kwargs={"classes": classes}, task_id=0)
    
    test_x = pickle.load(open("output/test_x.p", "rb"))
    probs = l2n.predict_proba(test_x, 0)
    pickle.dump(classes, open("output/classes100.p", "wb"))
    pickle.dump(probs, open("output/probs100.p", "wb"))


def compute_posteriors(test_x, test_y, l2n, num_tasks=10):

    # which task's test set.
    for s in range(num_tasks):
        # which task's posterior predictor.
        for t in range(num_tasks):
            probs_st = l2n.predict_proba(test_x[s], t)
            if t == 0:
                probs_t = probs_st
            else:
                probs_t = np.concatenate((probs_t, probs_st), axis=1)
        if s == 0:
            probs = probs_t
        else:
            probs = np.concatenate((probs, probs_t), axis=0)

    # Save test data again just to make sure posteriors match.
    pickle.dump(test_x, open("output/test_x.p", "wb"))
    pickle.dump(test_y, open("output/test_y.p", "wb"))
    pickle.dump(probs, open("output/probs.p", "wb"))

