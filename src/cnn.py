import numpy as np
import random
import matplotlib.pyplot as plt
import time
import cPickle as pickle
import os
import math


def relu(a):
    # gradient checking assert
    # assert(np.all(np.abs(a) > 1e-5))
    return a * (a > 0)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


# X = seq_len x batch_size
# E = vocab_size x embed_size
# F = list of filters of size filter_size x embed_size x num_filters
# B = list of filter biases of size 1 x 1 x num_filters
def conv_forward_prop(X, E, F, B, keep_prob):
    seq_len, batch_size = X.shape

    X = X.T  # batch_size x seq_len
    XE = E[X, :]  # batch_size x seq_len x embed_size

    features, M_idxs, iss, jss, CAs, XE_cols, Ws = [], [], [], [], [], [], []
    for f, b in zip(F, B):
        filter_size, embed_size, num_filters = f.shape
        out_height = seq_len - filter_size + 1

        i = np.repeat(np.arange(filter_size), embed_size).reshape(1, -1) + np.repeat(np.arange(out_height), 1).reshape(-1, 1)
        j = np.array([np.tile(np.arange(embed_size), filter_size)] * out_height)
        XE_col = XE[:, i, j]  # batch_size x out_height x (filter_size * embed_size)
        XE_col = XE_col.reshape((XE_col.shape[0] * XE_col.shape[1], XE_col.shape[2]))  # (batch_size * out_height) x (filter_size * embed_size)
        W = f.reshape(f.shape[0] * f.shape[1], f.shape[2])  # (filter_size * embed_size) x num_filters

        # convolution
        CZ = np.dot(XE_col, W)  # (batch_size * out_height) x num_filters
        CZ = CZ.reshape((batch_size, out_height, num_filters))
        CZ += b

        # activation
        CA = relu(CZ)

        # max-pool
        features += [np.amax(CA, axis=1)]
        M_idxs += [np.argmax(CA, axis=1)]

        # cache
        iss += [i]
        jss += [j]
        CAs += [CA]
        XE_cols += [XE_col]
        Ws += [W]

    A = np.concatenate(features, axis=1).T
    D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
    A *= D
    A /= keep_prob

    cache = features, M_idxs, iss, jss, CAs, XE_cols, Ws, XE, X, E, F, D, keep_prob
    return A, cache


# dA = total_filters x batch_size
def conv_backward_prop(dA, cache):

    features, M_idxs, iss, jss, CAs, XE_cols, Ws, XE, X, E, F, D, keep_prob = cache
    batch_size, seq_len = X.shape
    vocab_size, embed_size = E.shape

    dA /= keep_prob
    dA *= D
    dA = dA.T  # batch_size x total_filters

    dFeatures = np.split(dA, np.cumsum([f.shape[1] for f in features][:-1]), axis=1)

    dF, dB = [], []
    dXE = np.zeros((batch_size, seq_len, embed_size))
    for f, dFeature, M_idx, i, j, CA, XE_col, W in zip(F, dFeatures, M_idxs, iss, jss, CAs, XE_cols, Ws):

        filter_size, embed_size, num_filters = f.shape
        out_height = seq_len - filter_size + 1

        ca_i = np.repeat(np.arange(M_idx.shape[0]), M_idx.shape[1])
        ca_j = np.tile(np.arange(M_idx.shape[1]), M_idx.shape[0])
        ca_k = M_idx.reshape(-1)

        dCA = np.zeros((batch_size, out_height, num_filters))
        dCA[ca_i, ca_k, ca_j] = dFeature[ca_i, ca_j]
        dCZ = relu_backward(dCA, CA)

        dB += [np.sum(dCZ, axis=(0, 1)) / batch_size]
        dCZ = dCZ.reshape((batch_size * out_height, num_filters))
        dXE_col = np.dot(dCZ, W.T)
        dW = np.dot(XE_col.T, dCZ)
        dF += [dW.reshape((f.shape[0], f.shape[1], f.shape[2])) / batch_size]
        dXE_col = dXE_col.reshape((batch_size, out_height, filter_size * embed_size))

        np.add.at(dXE, (slice(None), i, j), dXE_col)  # this line takes the most time in backprop

    dE = np.zeros((vocab_size, embed_size))
    X = X.reshape(-1)
    dXE = dXE.reshape((dXE.shape[0] * dXE.shape[1], dXE.shape[2]))

    np.add.at(dE, (X, slice(None)), dXE)
    dE /= batch_size

    return dE, dF, dB


# X = prev_layer_size x batch_size
# W = curr_layer_size x prev_layer_size
# b = prev_layer_size x 1
def regular_forward_prop(X, W, b, activation, keep_prob):
    Z = np.dot(W, X) + b
    A = activation(Z)
    D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
    A *= D
    A /= keep_prob

    cache = X, W, A, D, keep_prob
    return A, cache


def relu_backward(dA, A):
    return dA * (A > 0)


def sigmoid_backward(dA, A):
    return dA * A * (1 - A)


# dA = next_layer_size x batch_size
def regular_backward_prop(dA, cache, activation_backward):
    X, W, A, D, keep_prob = cache
    batch_size = X.shape[1]

    dA /= keep_prob
    dA *= D
    dZ = activation_backward(dA, A)
    db = np.sum(dZ, axis=1, keepdims=True) / batch_size
    dW = np.dot(dZ, X.T) / batch_size
    dX = np.dot(W.T, dZ)

    return dX, dW, db


def random_split_batch(X_train, y_train, mini_batch_size):
    total_batch_size = X_train.shape[1]

    perm = np.random.permutation(total_batch_size)
    full_mini_batches = int(np.floor(1.0 * total_batch_size / mini_batch_size))

    mini_batches = []
    for i in range(full_mini_batches):
        X = X_train[:, perm[i*mini_batch_size:(i+1)*mini_batch_size]]
        y = y_train[:, perm[i*mini_batch_size:(i+1)*mini_batch_size]]
        mini_batches.append((X, y))

    if total_batch_size % mini_batch_size != 0:
        X = X_train[:, perm[full_mini_batches*mini_batch_size:total_batch_size]]
        y = y_train[:, perm[full_mini_batches*mini_batch_size:total_batch_size]]
        mini_batches.append((X, y))

    return mini_batches


def J(X, y, params, keep_probs):
    E, F, b, W1, b1, W2, b2 = params

    batch_size = X.shape[1]

    A0, conv_cache = conv_forward_prop(X, E, F, b, keep_probs[0])
    A1, regular_cache1 = regular_forward_prop(A0, W1, b1, relu, keep_probs[1])
    A2, regular_cache2 = regular_forward_prop(A1, W2, b2, sigmoid, 1.0)

    cost = np.sum((-y * np.log(A2) - (1 - y) * np.log(1 - A2)), axis=1) / batch_size

    caches = conv_cache, regular_cache1, regular_cache2
    return cost, A2, caches


def roll(params):
    rolled = np.array([])
    for param in params:
        rolled = np.append(rolled, param.reshape(-1))

    return rolled


def params2tuple(params, total_filters):
    E = params[0]
    F = params[1:1+total_filters]
    b = params[1+total_filters:1+2*total_filters]
    W1 = params[1+2*total_filters]
    b1 = params[2+2*total_filters]
    W2 = params[3+2*total_filters]
    b2 = params[4+2*total_filters]

    return E, F, b, W1, b1, W2, b2


def unroll(rolled, params, total_filters):
    unrolled = [None] * len(params)
    start = 0
    for i in range(len(params)):
        unrolled[i] = rolled[start:start+params[i].size].reshape(params[i].shape)
        start += params[i].size

    return params2tuple(unrolled, total_filters)


def gradient_checking(params, grads, X, y, total_filters):
    r_params = roll(params)
    r_params = r_params.astype(np.float128)

    J_plus, J_minus = np.zeros((len(r_params))), np.zeros((len(r_params)))
    print("len of r_params = " + str(len(r_params)))
    for i in range(len(r_params)):
        original = r_params[i]
        r_params[i] = original + 1e-5
        J_plus[i], _, _ = J(X, y, unroll(r_params, params, total_filters), [1.0, 1.0])
        r_params[i] = original - 1e-5
        J_minus[i], _, _ = J(X, y, unroll(r_params, params, total_filters), [1.0, 1.0])
        r_params[i] = original

    d_theta = roll(grads)
    d_theta_approx = (J_plus - J_minus) / 2 / 1e-5

    error = np.linalg.norm(d_theta - d_theta_approx) / (np.linalg.norm(d_theta) + np.linalg.norm(d_theta_approx))
    print("error = " + str(error))

    return


def calc_accuracy(A, Y):
    predictions = A > 0.5
    return 1.0 * np.sum(Y * predictions + (1 - Y) * (1 - predictions)) / Y.size


def random_initialization(vocab_size, embedding_size, num_filters, filter_sizes, hidden_units):
    total_filters = len(filter_sizes)

    E = np.random.rand(vocab_size, embedding_size) * 2 - 1
    F = [np.random.randn(filter_size, embedding_size, num_filters) * np.sqrt(6.0 / filter_size / embedding_size) for filter_size in filter_sizes]
    b = [np.zeros((1, 1, num_filters)) for i in range(total_filters)]
    W1 = np.random.randn(hidden_units, num_filters * total_filters) * np.sqrt(2.0 / num_filters * total_filters)
    b1 = np.zeros((hidden_units, 1))
    W2 = np.random.randn(1, hidden_units) * np.sqrt(1.0 / hidden_units)
    b2 = np.zeros((1, 1))

    return [E] + F + b + [W1, b1, W2, b2]


# X_train = seq_len x batch_size
# y_train = 1 x batch_size
def cnn(X_train, y_train, X_dev, y_dev, load_params_file, dump_dir, vocab_size, embedding_size,
        num_filters, filter_sizes, hidden_units, num_epochs, mini_batch_size, alpha, beta1, beta2,
        epsilon, keep_probs, plot_cost=True):

    np.random.seed(7)
    random.seed(7)
    total_filters = len(filter_sizes)

    if load_params_file is None:
        params = random_initialization(vocab_size, embedding_size, num_filters, filter_sizes, hidden_units)
        v_grads = [0] * len(params)
        s_grads = [0] * len(params)
        iteration = 0
        start_epoch = 0
        costs = []
    else:
        params, v_grads, s_grads, costs, iteration, start_epoch = pickle.load(open(load_params_file, "rb"))

    hyperparams = {
        "load_params_file": load_params_file, "dump_dir": dump_dir, "vocab_size": vocab_size,
        "embedding_size": embedding_size, "num_filters": num_filters, "filter_sizes": filter_sizes,
        "hidden_units": hidden_units, "num_epochs": num_epochs, "mini_batch_size": mini_batch_size,
        "alpha": alpha, "beta1": beta1, "beta2": beta2, "epsilon": epsilon, "keep_probs": keep_probs,
        "plot_cost": plot_cost, "iteration": iteration, "start_epoch": start_epoch
    }
    pickle.dump(hyperparams, open(os.path.join(dump_dir, "hyperparams.txt"), "wb"))

    print("iteration = %s start_epoch = %s" % (iteration, start_epoch))
    for epoch in range(start_epoch, num_epochs):
        mini_batches = random_split_batch(X_train, y_train, mini_batch_size)

        epoch_cost = 0
        epoch_accuracy = 0
        for mini_batch in mini_batches:
            iteration += 1

            # if iteration % 5 == 0:
            #     break

            X, y = mini_batch

            cost, A2, caches = J(X, y, params2tuple(params, total_filters), keep_probs)
            conv_cache, regular_cache1, regular_cache2 = caches

            train_accuracy = calc_accuracy(A2, y)
            logging_data = "iteration = %s cost = %s train_accuracy = %s" % (iteration, cost, train_accuracy)
            print(logging_data)
            pickle.dump(logging_data, open(os.path.join(dump_dir, "log.txt"), "ab"))

            if math.isnan(cost):
                return

            epoch_cost += cost
            epoch_accuracy += train_accuracy

            dA2 = -y / A2 + (1 - y) / (1 - A2)
            dA1, dW2, db2 = regular_backward_prop(dA2, regular_cache2, sigmoid_backward)
            dA0, dW1, db1 = regular_backward_prop(dA1, regular_cache1, relu_backward)
            dE, dF, db = conv_backward_prop(dA0, conv_cache)

            grads = [dE] + dF + db + [dW1, db1, dW2, db2]

            # gradient_checking(params, grads, X, y, total_filters)

            v_grads = [v * beta1 + g * (1 - beta1) for v, g in zip(v_grads, grads)]
            s_grads = [s * beta2 + g * g * (1 - beta2) for s, g in zip(s_grads, grads)]

            v_grads_corrected = [v / (1 - np.power(beta1, iteration)) for v in v_grads]
            s_grads_corrected = [s / (1 - np.power(beta2, iteration)) for s in s_grads]

            params = [p - alpha * v / (np.sqrt(s) + epsilon) for p, v, s in zip(params, v_grads_corrected, s_grads_corrected)]

        epoch_cost /= len(mini_batches)
        epoch_accuracy /= len(mini_batches)

        costs.append(epoch_cost)

        cost_dev, A2_dev, _ = J(X_dev, y_dev, params2tuple(params, total_filters), [1.0, 1.0])
        dev_accuracy = calc_accuracy(A2_dev, y_dev)

        logging_data = "epoch = %s epoch_cost = %f alpha = %f epoch_accuracy = %f dev_accuracy = %f" % \
                       (epoch, epoch_cost, alpha, epoch_accuracy, dev_accuracy)
        pickle.dump(logging_data, open(os.path.join(dump_dir, "log.txt"), "ab"))

        training_data = [params, v_grads, s_grads, costs, iteration, epoch+1]
        pickle.dump(training_data, open(os.path.join(dump_dir, "training_" + str(epoch) + ".txt"), "wb"))

        print("cost after epoch %i: %f" % (epoch, epoch_cost))
        print("alpha = " + str(alpha))
        print("train epoch accuracy = " + str(epoch_accuracy))
        print("dev accuracy = " + str(dev_accuracy))

    if plot_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(alpha))
        plt.show()

    return params
