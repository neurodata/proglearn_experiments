from utils import load_data, run_exp

train_x, train_y, test_x, test_y = load_data(split=False, task_prior=True)
run_exp(train_x, train_y, test_x, test_y, lr=0.003, epochs = 20, name="task_prior")

