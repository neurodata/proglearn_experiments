# from utils import pull_data,
from utils import load_data, fit_model, compute_posteriors

# Run experiment.

# If first time, run pull_data to get the CIFAR 100 dataset.
# pull_data()

train_x, train_y, test_x, test_y = load_data()
l2n = fit_model(train_x, train_y)


# Check accuracy on two tasks.

# Posteriors are of the form [n_test * 100],
# where the first 10 columns is the psoterior
# from task 1, next 10 for task 2, etc.
compute_posteriors(test_x, test_y, l2n)

