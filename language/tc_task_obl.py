from utils import load_imdb, load_yelp, load_amazon
from task_obl_utils import fit_source_tasks, compute_posteriors

# Experimental parameters.
n_estimators = 10
verbose = True

# Source tasks.
# TODO fill subsample fracs with actual values.
source_tasks = [
    {
        "name": "Yelp Review Sentiment Analysis",
        "filename": "yelp",
        "load": load_yelp,
        "subsample_frac": 0.01,
        "id": 0,
    },
    {
        "name": "IMDB Review Sentiment Analysis",
        "filename": "imdb",
        "load": load_imdb,
        "subsample_frac": 0.01,
        "id": 1,
    },
    {
        "name": "Amazon Review Sentiment Analysis",
        "filename": "amazon",
        "load": load_amazon,
        "subsample_frac": 0.01,
        "id": 2,
    },
]

l2f = fit_source_tasks(source_tasks, n_estimators=n_estimators, verbose=verbose)
compute_posteriors(l2f, source_tasks, verbose=verbose)
