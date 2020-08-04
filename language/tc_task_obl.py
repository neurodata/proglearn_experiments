from utils import get_source_and_target
from task_obl_utils import fit_source_tasks, compute_posteriors

# Experimental parameters.
n_estimators = 35
verbose = True

# Source tasks.
source_names = ["yelp", "imdb", "amazon"]
target_name = "toxic"
multitask_id = 5

source_tasks, target_task = get_source_and_target(source_names, target_name)
fit_source_tasks(source_tasks, multitask_id, n_estimators=n_estimators, verbose=verbose)
compute_posteriors(
    source_tasks, target_task, multitask_id, n_estimators=n_estimators, verbose=verbose,
)

