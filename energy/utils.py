import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from proglearn.network import LifelongRegressionNetwork


def load_data(city_name, verbose=False):
    
    # Load pre-COVID-19 (Feb 15) data. Some start from Jan 1 2018, and other from Jan 1 2019.
    data = np.genfromtxt('Data_Processed_New/Data_wo_Mobi/%s_data_all.csv' % city_name, delimiter=',')
    X = data[:, 1:].astype(float)
    y = data[:, 0].astype(float)
    
    if verbose:
        print(city_name, "Energy Consumption Data:")
        print("Sample size n =", X.shape[0])
        print("Number of features d =", X.shape[1])
    
    return X, y


def init_network(input_dim):
    
    model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(input_dim,), name='input'))
    # model.add(Dropout(0.2, name='dp0'))
    # model.add(Dense(128, activation='relu',  input_shape=(input_dim,), name='fc1'))
    # model.add(Dropout(0.2, name='dp0'))
    model.add(Dense(32, activation='relu', input_shape=(input_dim,), name='fc2'))
    model.add(Dense(16, activation='relu', name='fc3'))
    model.add(Dense(1, activation='linear', name='output'))
    
    return model


def fit_network(X_train, y_train, **kwargs):

    input_dim = X_train.shape[1]
    model = init_network(input_dim)
    model.compile(loss='mape', metrics=["mae", "mape"], optimizer=Adam(lr))
    model.fit(X_train, y_train, epochs=epochs, verbose=verbose)

    return model


def init_model(X_train, y_train, algo):
    
    if algo['filename'] == 'fixed_nn' or algo['filename'] == 'rolling_nn':
        return fit_network(X_train, y_train, **algo['fit_kwargs'])
    elif 'l2n' in algo['filename']: 
        input_dim = X_train.shape[1]
        l2n = LifelongRegressionNetwork(init_network(input_dim), **algo['init_kwargs'])

        num_tasks = algo['num_tasks']
        for j in range(num_tasks):
            n_per_task = len(X_train) // num_tasks
            X_task = X_train[j * n_per_task : (j + 1) * n_per_task, :]
            y_task = y_train[j * n_per_task : (j + 1) * n_per_task]
            l2n.add_task(X_task, y_task, task_id=j)

        return l2n
    else:
        raise ValueError("Unrecognized algorithm!")


def fit_model(X, y, t, n_train, algo):
    if algo['filename'] == 'fixed_nn':
        return algo
    elif algo['filename'] == 'rolling_nn':
        if t - algo['last_retrain'] >= algo['retrain_interval']:
            print("Retraining at index t=%d out of %d" % (t, len(X)))
            X_train = X[t - n_train:t, :]
            y_train = y[t - n_train:t]
            algo['model'] = fit_network(X_train, y_train, **algo['fit_kwargs'])
            algo['last_retrain'] = t
            return algo
        return algo
    elif 'l2n' in algo['filename']:
        if t - algo['last_retrain'] >= n_train // algo['num_tasks']:
            # We have seen a new "task". retrain progressively.
            print("Retraining PL, index t=%d out of %d" % (t, len(X)))
            X_train = X[t - n_train:t, :]
            y_train = y[t - n_train:t]
            
            algo['model'] = init_model(X_train, y_train, algo)
            algo['last_retrain'] = t
        return algo
    else:
        raise ValueError("Unrecognized algorithm!")


def compute_mape(X_test, y_test, algo):
    model = algo['model']
    if 'l2n' in algo['filename']:
        y_pred = model.predict(X_test, algo['num_tasks'] - 1)
    else:
        y_pred = model.predict(X_test)
    return np.mean(np.abs((y_test.reshape(-1, 1) - y_pred) / y_test))


def compute_rolling_mape(X, y, n_train, stride, algo, n_test=200):    
    # t indexes the timepoint at which we compute relative error.
    timepoints = []
    errors = []
    
    N = len(X)
    t = n_train
    while t + n_test < N:
    
        X_test, y_test = X[t:t + n_test, :], y[t:t + n_test]
        
        algo = fit_model(X, y, t, n_train, algo)
        error = compute_mape(X_test, y_test, algo)
        
        timepoints.append(t)
        errors.append(error)
        
        t += stride
    
    result = {
        'timepoints' : timepoints,
        'errors' : errors
    }

    pickle.dump(result, open("output/%s.pkl" % algo['filename'], "wb"))


def plot_errors(algos, city_name):
    sns.set(font_scale=1)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['figure.figsize'] = [10, 7]
    
    fig, ax = plt.subplots(1, 1)
    
    # key = model name, value = errors
    for algo in algos:
        result = pickle.load(open("output/%s.pkl" % algo['filename'], "rb"))
        ax.plot(result['timepoints'],
                result['errors'], 
                label=algo['name'], 
                linewidth=1.5, 
                color=algo['color'])
 
    # See 5% baseline.
    upper = 0.3
    ax.set_ylim((-0.05 * upper, upper))
    ax.axhline(y=0.05, linestyle='--', color='k')
    ax.axhline(y=0.10, linestyle='--', color='r')
    
    ax.set_ylabel("Mean Absolute Percentage Error")
    ax.set_xlabel("Time")
    
    # Important dates.
    date_labels = ["Jan 1, 2020", "Feb 15, 2020"]
    dates = [365 * 24 - 1, (365 + 31 + 15) * 24 - 1]
    for date in dates:
        ax.axvline(x=date, linestyle='solid', color='k', linewidth=0.75)
        
    ax.set_xticks(dates)
    ax.set_xticklabels(date_labels, rotation=45)
    
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_yticklabels(["0", "Baseline 5%", "Anomaly 10%"])
    
    ax.set_title("%s Load Forecasting Performance over Time" % city_name)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("%s_error_plot.pdf" % city_name, bbox_inches="tight")
    plt.show()
