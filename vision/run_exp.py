# from utils import pull_data,
from utils import load_data, fit_model, compute_posteriors

# Run experiment.

# If first time, run pull_data to get the CIFAR 100 dataset.
# pull_data()

train_x, train_y, test_x, test_y = load_data()
l2n = fit_model(train_x, train_y, lr = 0.003, batch_size=64)

# from sklearn.metrics import accuracy_score

# acc = accuracy_score(l2n.predict(test_x[0], 0), test_y[0])
# print("Accuracy:", acc)
# Check accuracy on two tasks.

# Posteriors are of the form [n_test * 100],
# where the first 10 columns is the psoterior
# from task 1, next 10 for task 2, etc.
compute_posteriors(test_x, test_y, l2n)

