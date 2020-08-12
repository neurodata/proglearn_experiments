import pickle

n_train = 500

train_idx = []
for t in range(10):
    train_idx_t = pickle.load(open("output/train_idx_small_%d.p" % t, "rb")) + t * 500
    train_idx.append(train_idx_t)

pickle.dump(train_idx, open("output/train_idx_task_prior_small.p", "wb"))
