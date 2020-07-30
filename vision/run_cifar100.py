from utils import pull_data, load_data, run_exp_100

# pull_data()
train_x, train_y, _, _ = load_data(split=False)
run_exp_100(train_x, train_y)
