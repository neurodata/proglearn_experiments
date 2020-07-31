from utils import get_source_tasks
from task_obl_utils import load_pooled_data, fit_task_priors, predict_task_priors

# Hyperparameters
n_estimators = 10

source_tasks = get_source_tasks()

X_train_pooled, t_train_pooled, X_test_pooled, t_test_pooled = load_pooled_data(
    source_tasks
)

# fit_task_priors(X_train_pooled, t_train_pooled, n_estimators=n_estimators)
predict_task_priors(X_test_pooled, t_test_pooled, n_estimators=n_estimators)

