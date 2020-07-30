import numpy as np
import pickle

from utils import load_imdb, load_yelp, load_amazon

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from proglearn.forest import UncertaintyForest

# Experimental parameters
n_estimators = 10
verbose = True
subsample_fracs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.999]
# subsample_fracs = [0.001, 0.003]  # for testing

source_tasks = [
    {
        'name' : 'Yelp Review Sentiment Analysis',
        'filename' : 'yelp',
        'load' : load_yelp,
    },
    {
        'name' : 'IMDB Review Sentiment Analysis',
        'filename' : 'imdb',
        'load' : load_imdb,
    },
    {
        'name' : 'Amazon Review Sentiment Analysis',
        'filename' : 'amazon',
        'load' : load_amazon,
    }
]

for task in source_tasks:
    print("----------------------------------------")
    print("TASK:", task['name'])    
    print("----------------------------------------")

    # Load all of the train and test data.
    X_train_full, y_train_full, X_test, y_test = task['load'](verbose = verbose)
    
    accs = np.zeros(len(subsample_fracs))
    for i, subsample_frac in enumerate(subsample_fracs):

        # Fit only on a fraction of the training data.
        _, X_train, _, y_train = train_test_split(X_train_full, y_train_full, test_size=subsample_frac)
        
        print("SUBSAMPLE FRAC:", subsample_frac)
        print("n_train:", len(X_train))

        uf = UncertaintyForest(n_estimators=n_estimators)
        uf.fit(X_train, y_train)
        
        # Compute accuracy of this learner.
        accs[i] = accuracy_score(uf.predict(X_test), y_test)
    
    pickle.dump(accs, open("output/uf_accs_%s_%d.p" % (task['filename'], n_estimators), "wb"))
    print("----------------------------------------")
pickle.dump(subsample_fracs, open("output/uf_subsample_fracs.p", "wb"))
