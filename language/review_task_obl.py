from task_obl_utils import run_zero_shot

# Hyperparameters
n_estimators = 10
verbose = True

# Multitask 0:
multitask_id = 0
source_names = ["yelp", "imdb"]
target_name = "amazon"

run_zero_shot(source_names, target_name, multitask_id, n_estimators, verbose)
