from utils import get_source_tasks
from task_obl_utils import fit_source_tasks, compute_posteriors

# Experimental parameters.
n_estimators = 10
verbose = True

# Source tasks.
source_tasks = get_source_tasks()
fit_source_tasks(source_tasks, n_estimators=n_estimators, verbose=verbose)
compute_posteriors(source_tasks, n_estimators=n_estimators, verbose=verbose)

