from data_helpers import load_data_and_labels
from tensorflow.contrib import learn
import numpy as np
import random
import cnn


x_text, y = load_data_and_labels("../data/rt-polaritydata/rt-polarity.pos", "../data/rt-polaritydata/rt-polarity.neg")
y = np.dot(y, [[0], [1]])

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
# max_document_length = 3
min_frequency = 0
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(1)
shuffle_indices = np.random.permutation(len(y))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/dev/test set
train_size = 8000
dev_size = (x.shape[0] - train_size) / 2
x_train, x_dev, x_test = x_shuffled[:train_size], x_shuffled[train_size:train_size+dev_size], x_shuffled[train_size+dev_size:]
y_train, y_dev, y_test = y_shuffled[:train_size], y_shuffled[train_size:train_size+dev_size], y_shuffled[train_size+dev_size:]

x_train, x_dev, x_test = x_train.T, x_dev.T, x_test.T
y_train, y_dev, y_test = y_train.T, y_dev.T, y_test.T

vocab_size = len(vocab_processor.vocabulary_)
embedding_size = 64
num_filters = 32
filter_sizes = [3, 4, 5]
hidden_units = 20
num_epochs = 100
mini_batch_size = 128
alpha = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# vocab_size = len(vocab_processor.vocabulary_)
# embedding_size = 2
# num_filters = 2
# filter_sizes = [2]
# hidden_units = 3
# num_epochs = 100
# mini_batch_size = 64
# alpha = 0.01
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# x_train = x_train[:, 7:8]
# y_train = y_train[:, 7:8]

# print(x_train)
# print(y_train)

print("vocab_size = " + str(vocab_size))

params = cnn.cnn(x_train, y_train, vocab_size, embedding_size, num_filters, filter_sizes, hidden_units, num_epochs, mini_batch_size, alpha, beta1, beta2, epsilon)

exit(0)

print(max_document_length)
print(x_text[0])
print(x[0])

idx = [0, 2, 0, 4]
X = [[0, 2], [0, 4], [1, 1]]
E = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])

EX = E[X, :]
print(EX.shape)
EX = EX.reshape((6, 4))
print(EX)
print(EX.shape)

seq_len = 4
filter_size = 3
C = np.arange(seq_len * filter_size).reshape(seq_len, filter_size)
print(C)
print(C[1:4, 0:2])

S = np.array([np.trace(C, x) for x in range(0, -(seq_len - filter_size) - 1, -1)])
print(S)

list = [
    [[1, 2], [3, 4]],
    [[5, 6]],
    [[7, 8], [9, 10], [11, 12]]
]

m = np.concatenate(list)
print(m)

a = np.array([1, 2, 5, 5, 2, 1])
m = np.amax(a)
indices = np.nonzero(a == m)[0]
print(random.choice(indices))
print(random.choice(indices))
print(random.choice(indices))

X = np.array([[0, 1], [2, 2], [0, 3], [1, 1]])
E = np.array([[1, 2], [5, 6], [9, 10], [13, 14]])
F = [
    np.array([[-1, -1], [-1, 0], [0, -1]]),
    np.array([[0, 0], [1, 1], [0, 0]]),
    np.array([[1, 1], [2, 2]]),
    np.array([[10, 10], [1000, 1000]])
]
b = [1, 0, 2, 3]

# features, cache = cnn.conv_forward_prop(X, E, F, b)
# dA = np.arange(features.shape[0] * features.shape[1]).reshape(features.shape)
# cnn.conv_backward_prop(dA, cache)


