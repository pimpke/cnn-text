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
# F = list of filter_size x embed_size
# b = list of filter biases
def conv_forward_prop(X, E, F, b):

    seq_len, batch_size = X.shape
    filter_sizes = [f.shape[0] for f in F]
    cum_sum_filter_sizes = np.cumsum(filter_sizes)

    X = X.T  # batch_size x seq_len
    XE = E[X, :]  # batch_size x seq_len x embed_size
    XE = XE.reshape((XE.shape[0] * XE.shape[1], XE.shape[2]))  # (batch_size * seq_len) x embed_size

    FC = np.concatenate(F)  # total_sum_filter_sizes x embed_size

    # i = np.repeat(np.arange(filter_size), embed_size).reshape(1, -1) + np.repeat(np.arange(seq_len - filter_size + 1), 1).reshape(-1, 1)
    # j = np.array([np.tile(np.arange(embed_size), filter_size)] * (seq_len - filter_size + 1))
    #
    # XE[:, i, j].reshape()

    start = time.time()
    C = np.dot(XE, FC.T)  # (batch_size * seq_len) x total_sum_filter_sizes
    print("the great matrix mul time = " + str(time.time() - start))
    print("XE.shape = " + str(XE.shape))
    print("C.shape = " + str(C.shape))
    print("FC.shape = " + str(FC.shape))

    start = time.time()
    features = np.zeros((cum_sum_filter_sizes.size, batch_size))
    M_idxs = np.zeros((batch_size, cum_sum_filter_sizes.size), dtype=np.int)
    should_pass_gradient = np.zeros((batch_size, cum_sum_filter_sizes.size))
    for batch_idx in range(batch_size):
        filter_idx = 0
        for filter_start, filter_end in zip(np.insert(cum_sum_filter_sizes, 0, 0), cum_sum_filter_sizes):

            curr = C[(batch_idx*seq_len):((batch_idx+1)*seq_len), filter_start:filter_end]
            S = np.array([np.trace(curr, x) for x in range(0, -(seq_len - filter_end + filter_start) - 1, -1)])

            S += b[filter_idx]
            S = relu(S)

            m = np.amax(S)
            m_idx = random.choice(np.nonzero(S == m)[0])

            features[filter_idx, batch_idx] = m
            M_idxs[batch_idx, filter_idx] = m_idx
            should_pass_gradient[batch_idx, filter_idx] = m > 0

            filter_idx += 1
    print("the 2 fors lolz = " + str(time.time() - start))

    cache = M_idxs, should_pass_gradient, cum_sum_filter_sizes, XE, FC, X, E
    return features, cache


# dA = total_filters x batch_size
def conv_backward_prop(dA, cache):

    M_idxs, should_pass_gradient, cum_sum_filter_sizes, XE, FC, X, E = cache
    batch_size, seq_len = X.shape
    vocab_size, embed_size = E.shape
    dC = np.zeros((batch_size * seq_len, FC.shape[0]))
    dB = np.zeros((batch_size, cum_sum_filter_sizes.size))
    dA = dA.T

    for batch_idx in range(batch_size):
        filter_idx = 0
        for filter_start, filter_end in zip(np.insert(cum_sum_filter_sizes, 0, 0), cum_sum_filter_sizes):
            if should_pass_gradient[batch_idx, filter_idx]:
                m = M_idxs[batch_idx, filter_idx]
                dC_rows = np.arange(batch_idx * seq_len + m, batch_idx * seq_len + m + filter_end - filter_start)
                dC_cols = np.arange(filter_start, filter_end)
                dC[dC_rows, dC_cols] = dA[batch_idx, filter_idx]
                dB[batch_idx, filter_idx] = dA[batch_idx, filter_idx]
            filter_idx += 1

    dXE = np.dot(dC, FC)
    dFC = np.dot(XE.T, dC).T

    dF = [dFC[filter_start:filter_end, :] for filter_start, filter_end in zip(np.insert(cum_sum_filter_sizes, 0, 0), cum_sum_filter_sizes)]
    for x in dF:
        x /= batch_size

    dE = np.zeros((vocab_size, embed_size))
    X = X.reshape((X.shape[0] * X.shape[1], 1))

    for word_idx in range(vocab_size):
        dE[word_idx, :] = np.sum(dXE[np.nonzero(X == word_idx)[0], :], axis=0, keepdims=True)
    dE /= batch_size

    db = np.sum(dB, axis=0, keepdims=True) / batch_size

    return dE, dF, db


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
    F = [np.random.randn(filter_size, embedding_size) * np.sqrt(6.0 / filter_size / embedding_size) for filter_size in filter_sizes for n in range(num_filters)]
    b = np.zeros((len(F)))
    W1 = np.random.randn(hidden_units, len(F)) * np.sqrt(2.0 / len(F))
    b1 = np.zeros((hidden_units, 1))
    W2 = np.random.randn(1, hidden_units) * np.sqrt(1.0 / hidden_units)
    b2 = np.zeros((1, 1))

    iteration = 0
    costs = []
    for epoch in range(num_epochs):
        mini_batches = random_split_batch(X_train, y_train, mini_batch_size)

        params = [E] + F + [b, W1, b1, W2, b2]
        v_grads = [0] * len(params)
        s_grads = [0] * len(params)
        epoch_cost = 0
        for mini_batch in mini_batches:
            start_iteration = time.time()
            iteration += 1

            X, y = mini_batch
            batch_size = X.shape[1]

            start = time.time()
            A0, conv_cache = conv_forward_prop(X, E, F, b)
            print("conv_forward time = " + str(time.time() - start))
            A1, regular_cache1 = regular_forward_prop(A0, W1, b1, relu)
            A2, regular_cache2 = regular_forward_prop(A1, W2, b2, sigmoid)

            cost = np.sum((-y * np.log(A2) - (1 - y) * np.log(1 - A2)), axis=1) / batch_size
            epoch_cost += cost

            dA2 = -y / A2 + (1 - y) / (1 - A2)
            dA1, dW2, db2 = regular_backward_prop(dA2, regular_cache2, sigmoid_backward)
            dA0, dW1, db1 = regular_backward_prop(dA1, regular_cache1, relu_backward)
            start = time.time()
            dE, dF, db = conv_backward_prop(dA0, conv_cache)
            print("conv_backward time = " + str(time.time() - start))

            grads = [dE] + dF + [db, dW1, db1, dW2, db2]

            # for i in range(len(v_grads)):
            #     print(i)
            #     print("v_grads type = " + type(v_grads[i]).__name__)
            #     print("grads type = " + type(grads[i]).__name__)
            #     # print("v_grads.shape = " + str(v_grads[i].shape))
            #     # print("grads.shape = " + str(grads[i].shape))
            #     tmp = v_grads[i] * beta1 + grads[i] * (1 - beta1)
            #     print(tmp)

            v_grads = [v * beta1 + g * (1 - beta1) for v, g in zip(v_grads, grads)]
            s_grads = [s * beta2 + g * g * (1 - beta2) for s, g in zip(s_grads, grads)]

            v_grads_corrected = [v / (1 - np.power(beta1, iteration)) for v in v_grads]
            s_grads_corrected = [s / (1 - np.power(beta2, iteration)) for s in s_grads]

            params = [p - alpha * v / (np.sqrt(s) + epsilon) for p, v, s in zip(params, v_grads_corrected, s_grads_corrected)]
            print("iteration time = " + str(time.time() - start_iteration))

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
