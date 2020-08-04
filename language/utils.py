import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split


def load_imdb(view="bert", verbose=False, subsample_frac=None, test_size=0.1):
    if view == "raw":
        df = pd.read_csv("data/IMDB/IMDB_Dataset.csv", sep=",")
        data = df.to_numpy()

        X, y = data[:, 0], data[:, 1]

        # Convert 'negative' and 'positive' label as 0 and 1 respectively.
        y = (y == "positive").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        if subsample_frac:
            _, X_train, _, y_train = train_test_split(
                X_train, y_train, test_size=subsample_frac, random_state=42
            )
        if verbose:
            print(
                "'X_train' and 'X_test' are each a list of string-valued reviews of movies."
            )
            print(
                "'y_train' and 'y_test' are each list of binary sentiments, where 0 = 'negative' and 1 = 'positive'."
            )
            print("Number of training examples =", len(X_train))
            print("Number of testing examples =", len(X_test))
            print("First few examples:")
            for i in range(3):
                print("Review: -------------------------------------------------------")
                print(X_train[i])
                print("Sentiment: ----------------------------------------------------")
                print(y_train[i])
    elif view == "bert":
        data = pickle.load(open("data/IMDB/imdb_google_bert.p", "rb"))
        X = np.array(data[0])
        y = data[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        if subsample_frac:
            _, X_train, _, y_train = train_test_split(
                X_train, y_train, test_size=subsample_frac, random_state=42
            )
        if verbose:
            print(
                "'X_train' and 'X_test' are each an n-by-d array of BERT embedded movie reviews."
            )
            print(
                "'y_train' and 'y_test' are each list of binary sentiments, where 0 = 'negative' and 1 = 'positive'."
            )
            print("Number of training examples =", len(X_train))
            print("Input dimension d =", X_train.shape[1])
            print("Number of testing examples =", len(X_test))
    else:
        raise ValueError("Unrecognized view!")

    return X_train, y_train, X_test, y_test


def load_yelp(view="bert", verbose=False, subsample_frac=None):
    if view == "raw":
        partition = {}
        for part in ["train", "test"]:
            df = pd.read_csv(
                "data/YELP/yelp_review_polarity_csv/%s.csv" % part, sep=",", header=None
            )
            data = df.to_numpy()
            partition["X_%s" % part] = data[:, 1]
            # Convert '1' (negative) and '2' (positive) label to 0 and 1, respectively.
            partition["y_%s" % part] = data[:, 0].astype(int) - 1

        X_train, y_train = partition["X_train"], partition["y_train"]
        X_test, y_test = partition["X_test"], partition["y_test"]

        if subsample_frac:
            _, X_train, _, y_train = train_test_split(
                X_train, y_train, test_size=subsample_frac, random_state=42
            )
        if verbose:
            print(
                "'X_train' and 'X_test' are each a list of string-valued reviews of business."
            )
            print(
                "'y_train' and 'y_test' are each list of binary sentiments, where 0 = 'negative' and 1 = 'positive'."
            )
            print("Number of training examples =", len(X_train))
            print("Number of testing examples =", len(X_test))
            print("First few examples:")
            for i in range(3):
                print("Review: -------------------------------------------------------")
                print(X_train[i])
                print("Sentiment: ----------------------------------------------------")
                print(y_train[i])
    elif view == "bert":
        partition = {}
        for part in ["train", "test"]:
            data = pickle.load(open("data/YELP/yelp_%s_google_bert.p" % part, "rb"))
            partition["X_%s" % part] = np.array(data[0])
            partition["y_%s" % part] = data[1]

        X_train, y_train = partition["X_train"], partition["y_train"]
        X_test, y_test = partition["X_test"], partition["y_test"]

        if subsample_frac:
            _, X_train, _, y_train = train_test_split(
                X_train, y_train, test_size=subsample_frac, random_state=42
            )
        if verbose:
            print(
                "'X_train' and 'X_test' are each an n-by-d array of BERT embedded reviews of a business."
            )
            print(
                "'y_train' and 'y_test' are each list of binary sentiments, where 0 = 'negative' and 1 = 'positive'."
            )
            print("Number of training examples =", len(X_train))
            print("Input dimension d =", X_train.shape[1])
            print("Number of testing examples =", len(X_test))
    else:
        raise ValueError("Unrecognized view!")

    return X_train, y_train, X_test, y_test


def load_amazon(
    view="bert", verbose=False, review_component="title", subsample_frac=None
):
    if view == "raw":
        partition = {}
        for part in ["train", "test"]:
            df = pd.read_csv(
                "data/Amazon/amazon_review_polarity_csv/%s.csv" % part,
                sep=",",
                header=None,
            )
            data = df.to_numpy()
            partition["X_%s" % part] = data[:, 1]
            # Convert '1' (negative) and '2' (positive) label to 0 and 1, respectively.
            partition["y_%s" % part] = data[:, 0].astype(float) - 1

        X_train, y_train = partition["X_train"], partition["y_train"]
        X_test, y_test = partition["X_test"], partition["y_test"]

        if subsample_frac:
            _, X_train, _, y_train = train_test_split(
                X_train, y_train, test_size=subsample_frac, random_state=42
            )
        if verbose:
            print(
                """'X_train' and 'X_test' are each 
                a list of string-valued reviews of products."""
            )
            print(
                """'y_train' and 'y_test' are each list of binary sentiments, 
                where 0 = 'negative' and 1 = 'positive'."""
            )
            print("Number of training examples =", len(X_train))
            print("Number of testing examples =", len(X_test))
            print("First few examples:")
            for i in range(3):
                print("Review: -------------------------------------------------------")
                print(X_train[i])
                print("Sentiment: ----------------------------------------------------")
                print(y_train[i])
    elif view == "bert":
        partition = {}
        for part in ["train", "test"]:
            data = pickle.load(
                open("data/Amazon/amazon_%s_sample_300000_google_bert.p" % part, "rb")
            )
            # The pickle is a tuple of (label, title_embedding, comment_embedding).
            partition["y_%s" % part] = data[0].astype(float) - 1
            if review_component == "title":
                partition["X_%s" % part] = np.array(data[1])
            elif review_component == "comment":
                partition["X_%s" % part] = np.array(data[2])
            else:
                raise ValueError("'review_component' must be 'title' or 'comment'")

        X_train, y_train = partition["X_train"], partition["y_train"]
        X_test, y_test = partition["X_test"], partition["y_test"]
        if subsample_frac:
            _, X_train, _, y_train = train_test_split(
                X_train, y_train, test_size=subsample_frac, random_state=42
            )
        if verbose:
            print(
                """'X_train' and 'X_test' are each an 
                n-by-d array of BERT embedded product reviews."""
            )
            print(
                """'y_train' and 'y_test' are each list of binary sentiments, 
                where 0 = 'negative' and 1 = 'positive'."""
            )
            print("Number of training examples =", len(X_train))
            print("Input dimension d =", X_train.shape[1])
            print("Number of testing examples =", len(X_test))
    else:
        raise ValueError("Unrecognized view!")

    return X_train, y_train, X_test, y_test


def load_toxic_comment(view="bert", verbose=False, subsample_frac=None):
    if view == "raw":

        # Training set has labels.
        df = pd.read_csv("data/toxic_comment/train.csv", sep=",")
        data = df.to_numpy()
        # First column is an ID.
        X_train = data[:, 1]
        y_train = data[:, 2:]

        # Test set has labels in a different file.
        df = pd.read_csv("data/toxic_comment/test.csv", sep=",")
        data = df.to_numpy()
        X_test = data[:, 1]

        df = pd.read_csv("data/toxic_comment/test_labels.csv", sep=",")
        data = df.to_numpy()
        y_test = data[:, 1:].astype(int)

        # Remove all examples that are unlabelled.
        unlabelled_examples = y_test[:, 0] != -1
        y_test = y_test[unlabelled_examples, :]
        X_test = X_test[unlabelled_examples]

        if subsample_frac:
            _, X_train, _, y_train = train_test_split(
                X_train, y_train, test_size=subsample_frac, random_state=42
            )
        if verbose:
            print(
                "'X_train' and 'X_test' are each a list of string-valued Wikipedia comments."
            )
            print(
                """'y_train' and 'y_test' are each list of multilabel binary sentiments, 
                where the columns indicate 'toxic', 'severe_toxic', 'obscene', 'threat', 
                'insult', 'identity_hate', and 'not_toxic', in that order."""
            )

            print("Number of training examples =", len(X_train))
            print("Number of testing examples =", len(X_test))

            print("First few examples:")
            classes = [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ]
            for i in range(3):
                print(
                    "Comment: -------------------------------------------------------"
                )
                print(X_train[i])
                print("Class: ----------------------------------------------------")
                print(classes[np.argmax(y_train[i])])
    elif view == "bert":

        # Train
        data = pickle.load(open("data/toxic_comment/toxic_train_google_bert.p", "rb"))
        tensors, ids = zip(*data[0])
        X_train = np.array(tensors)
        y_train = data[1].to_numpy()[:, 1:].astype(float)

        # X_test
        data = pickle.load(
            open("data/toxic_comment/toxic_test_google_bert_nolabels.p", "rb")
        )
        tensors, ids = zip(*data)
        X_test = np.array(tensors)

        # y_test
        df = pd.read_csv("data/toxic_comment/test_labels.csv", sep=",")
        data = df.to_numpy()
        y_test = data[:, 1:].astype(float)

        # Remove all examples that are unlabelled.
        unlabelled_examples = y_test[:, 0] != -1
        y_test = y_test[unlabelled_examples, :]
        X_test = X_test[unlabelled_examples]

        if subsample_frac:
            _, X_train, _, y_train = train_test_split(
                X_train, y_train, test_size=subsample_frac, random_state=42
            )
        if verbose:
            print(
                "'X_train' and 'X_test' are each an n-by-d array of BERT embedded reviews of a business."
            )
            print(
                """'y_train' and 'y_test' are each list of multilabel binary sentiments, 
                where the columns indicate 'toxic', 'severe_toxic', 'obscene', 'threat', 
                'insult', 'identity_hate', and 'not_toxic', in that order."""
            )
            print("Number of training examples =", len(X_train))
            print("Input dimension d =", X_train.shape[1])
            print("Number of testing examples =", len(X_test))
    else:
        raise ValueError("Unrecognized view!")

    return X_train, y_train, X_test, y_test


def get_source_and_target(
    source_names,
    target_name,
    sub_yelp=0.001,
    sub_imdb=0.01,
    sub_amazon=0.001,
    sub_toxic_comment=0.001,
):
    tasks = {
        "yelp": {
            "name": "Yelp Review Sentiment Analysis",
            "filename": "yelp",
            "load": load_yelp,
            "subsample_frac": sub_yelp,
            "id": 0,
        },
        "imdb": {
            "name": "IMDB Review Sentiment Analysis",
            "filename": "imdb",
            "load": load_imdb,
            "subsample_frac": sub_imdb,
            "id": 1,
        },
        "amazon": {
            "name": "Amazon Review Sentiment Analysis",
            "filename": "amazon",
            "load": load_amazon,
            "subsample_frac": sub_amazon,
            "id": 2,
        },
        "toxic_comment": {
            "name": "Toxic Comment Identification",
            "filename": "toxic_comment",
            "load": load_toxic_comment,
            "subsample_frac": sub_toxic_comment,
            "id": 3,
        },
    }

    source_tasks = [tasks[source_name] for source_name in source_names]
    target_task = tasks[target_name]

    return source_tasks, target_task

