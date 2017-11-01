import numpy as np
import random
import matplotlib.pyplot as plt
import time


def relu(a):
    return a * (a > 0)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


# X = seq_len x batch_size
# E = vocab_size x embed_size
# F = list of filters of size filter_size x embed_size x num_filters
# B = list of filter biases of size 1 x 1 x num_filters
def conv_forward_prop(X, E, F, B):
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

    a = np.concatenate(features, axis=1).T

    cache = features, M_idxs, iss, jss, CAs, XE_cols, Ws, XE, X, E, F
    return a, cache


# dA = total_filters x batch_size
def conv_backward_prop(dA, cache):

    dA = dA.T  # batch_size x total_filters
    features, M_idxs, iss, jss, CAs, XE_cols, Ws, XE, X, E, F = cache
    batch_size, seq_len = X.shape
    vocab_size, embed_size = E.shape

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
def regular_forward_prop(X, W, b, activation):
    Z = np.dot(W, X) + b
    A = activation(Z)

    cache = X, W, A
    return A, cache


def relu_backward(dA, A):
    return dA * (A > 0)


def sigmoid_backward(dA, A):
    return dA * A * (1 - A)


# dA = next_layer_size x batch_size
def regular_backward_prop(dA, cache, activation_backward):
    X, W, A = cache
    batch_size = X.shape[1]

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


# X_train = seq_len x batch_size
# y_train = 1 x batch_size
def cnn(X_train, y_train, vocab_size, embedding_size, num_filters, filter_sizes, hidden_units, num_epochs, mini_batch_size, alpha, beta1, beta2, epsilon, print_cost=True, plot_cost=True):

    np.random.seed(7)
    random.seed(7)

    E = np.random.rand(vocab_size, embedding_size) * 2 - 1
    F = [np.random.randn(filter_size, embedding_size, num_filters) * np.sqrt(6.0 / filter_size / embedding_size) for filter_size in filter_sizes]
    b = [np.zeros((1, 1, num_filters)) for i in range(len(filter_sizes))]
    W1 = np.random.randn(hidden_units, num_filters * len(filter_sizes)) * np.sqrt(2.0 / num_filters * len(filter_sizes))
    b1 = np.zeros((hidden_units, 1))
    W2 = np.random.randn(1, hidden_units) * np.sqrt(1.0 / hidden_units)
    b2 = np.zeros((1, 1))

    iteration = 0
    costs = []
    for epoch in range(num_epochs):
        mini_batches = random_split_batch(X_train, y_train, mini_batch_size)

        params = [E] + F + b + [W1, b1, W2, b2]
        v_grads = [0] * len(params)
        s_grads = [0] * len(params)
        epoch_cost = 0
        for mini_batch in mini_batches:
            # start_iteration = time.time()
            # print("iteration = " + str(iteration) + " of " + str(len(mini_batches)))
            iteration += 1

            X, y = mini_batch
            batch_size = X.shape[1]

            # start = time.time()
            A0, conv_cache = conv_forward_prop(X, E, F, b)
            # print("conv_forward time = " + str(time.time() - start))
            A1, regular_cache1 = regular_forward_prop(A0, W1, b1, relu)
            A2, regular_cache2 = regular_forward_prop(A1, W2, b2, sigmoid)

            cost = np.sum((-y * np.log(A2) - (1 - y) * np.log(1 - A2)), axis=1) / batch_size
            epoch_cost += cost
            print("iteration = " + str(iteration) + " cost = " + str(cost))

            dA2 = -y / A2 + (1 - y) / (1 - A2)
            dA1, dW2, db2 = regular_backward_prop(dA2, regular_cache2, sigmoid_backward)
            dA0, dW1, db1 = regular_backward_prop(dA1, regular_cache1, relu_backward)
            # start = time.time()
            dE, dF, db = conv_backward_prop(dA0, conv_cache)
            # print("conv_backward time = " + str(time.time() - start))

            grads = [dE] + dF + db + [dW1, db1, dW2, db2]

            v_grads = [v * beta1 + g * (1 - beta1) for v, g in zip(v_grads, grads)]
            s_grads = [s * beta2 + g * g * (1 - beta2) for s, g in zip(s_grads, grads)]

            v_grads_corrected = [v / (1 - np.power(beta1, iteration)) for v in v_grads]
            s_grads_corrected = [s / (1 - np.power(beta2, iteration)) for s in s_grads]

            params = [p - alpha * v / (np.sqrt(s) + epsilon) for p, v, s in zip(params, v_grads_corrected, s_grads_corrected)]
            # print("iteration time = " + str(time.time() - start_iteration))

        epoch_cost /= len(mini_batches)
        if print_cost: #and epoch % 100 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost: #and epoch % 5 == 0:
            costs.append(epoch_cost)

    if plot_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(alpha))
        plt.show()

    return params
