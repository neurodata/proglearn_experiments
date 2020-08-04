from task_obl_utils import run_zero_shot, run_task_aware_pl, run_single_task

# Hyperparameters
n_estimators = 10
verbose = True

# Multitask 0:
multitask_id = 0
print("MULTITASK 0: IMDB, AMAZON ---> YELP")
source_names = ["imdb", "amazon"]
target_name = "yelp"

run_zero_shot(source_names, target_name, 0, n_estimators, verbose)
run_task_aware_pl(source_names, target_name, 10, n_estimators, verbose)
run_single_task(target_name, 20, n_estimators, verbose)


print("MULTITASK: YELP, AMAZON ---> IMDB")
source_names = ["yelp", "amazon"]
target_name = "imdb"

run_zero_shot(source_names, target_name, 1, n_estimators, verbose)
run_task_aware_pl(source_names, target_name, 11, n_estimators, verbose)
run_single_task(target_name, 21, n_estimators, verbose)

print("MULTITASK: YELP, IMDB ---> AMAZON")
source_names = ["yelp", "imdb"]
target_name = "amazon"

run_zero_shot(source_names, target_name, 2, n_estimators, verbose)
run_task_aware_pl(source_names, target_name, 12, n_estimators, verbose)
run_single_task(target_name, 22, n_estimators, verbose)
