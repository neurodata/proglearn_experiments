from task_obl_utils import run_zero_shot

# Hyperparameters
n_estimators = 10
verbose = True

# Multitask 0:
multitask_id = 0
print("MULTITASK 0: IMDB, AMAZON ---> YELP")
source_names = ["imdb", "amazon"]
target_name = "yelp"

run_zero_shot(source_names, target_name, multitask_id, n_estimators, verbose)
