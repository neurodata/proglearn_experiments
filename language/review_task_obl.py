from task_obl_utils import run_zero_shot

# Hyperparameters
n_estimators = 10
verbose = True

# Multitask 0:
# multitask_id = 0
# print("MULTITASK 0: IMDB, AMAZON ---> YELP")
# source_names = ["imdb", "amazon"]
# target_name = "yelp"

# run_zero_shot(source_names, target_name, multitask_id, n_estimators, verbose)

# Multitask 1:
multitask_id = 1
print("MULTITASK 1: YELP, AMAZON ---> IMDB")
source_names = ["yelp", "amazon"]
target_name = "imdb"

run_zero_shot(source_names, target_name, multitask_id, n_estimators, verbose)

# Multitask 2:
multitask_id = 2
print("MULTITASK 2: YELP, IMDB ---> AMAZON")
source_names = ["yelp", "imdb"]
target_name = "amazon"

run_zero_shot(source_names, target_name, multitask_id, n_estimators, verbose)
