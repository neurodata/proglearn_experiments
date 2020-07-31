from utils import load_data, run_exp

train_x, train_y, test_x, test_y = load_data(split=False, task_prior=True)
run_exp(train_x, train_y, test_x, test_y, name="task_prior")

