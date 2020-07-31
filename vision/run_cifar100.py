from utils import pull_data, load_data, run_exp

# pull_data()
train_x, train_y, test_x, test_y = load_data(split=False)
run_exp(train_x, train_y, test_x, test_y, epochs = 50, lr = 0.001, name='cifar100')
