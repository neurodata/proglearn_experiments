from utils import get_source_tasks
from task_obl_utils import fit_source_tasks, compute_posteriors

# Experimental parameters.
n_estimators = 35
verbose = True

# Source tasks.
source_tasks = get_source_tasks(sub_yelp = 0.01, sub_imdb = 0.1, sub_amazon = 0.01)
fit_source_tasks(source_tasks, n_estimators=n_estimators, verbose=verbose)
compute_posteriors(source_tasks, n_estimators=n_estimators, verbose=verbose)

