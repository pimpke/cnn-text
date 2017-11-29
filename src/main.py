from data_helpers import load_data_and_labels
from tensorflow.contrib import learn
import numpy as np
import random
import cnn
import cPickle as pickle
import os

x_text, y = load_data_and_labels("../data/rt-polaritydata/rt-polarity.pos", "../data/rt-polaritydata/rt-polarity.neg")
y = np.dot(y, [[0], [1]])

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(1)
shuffle_indices = np.random.permutation(len(y))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/dev/test set
train_size = 8000
dev_size = (x.shape[0] - train_size) / 2

# dev_size = x.shape[0] / 10
# train_size = x.shape[0] - dev_size

x_train, x_dev, x_test = x_shuffled[:train_size], x_shuffled[train_size:train_size+dev_size], x_shuffled[train_size+dev_size:]
y_train, y_dev, y_test = y_shuffled[:train_size], y_shuffled[train_size:train_size+dev_size], y_shuffled[train_size+dev_size:]

x_train, x_dev, x_test = x_train.T, x_dev.T, x_test.T
y_train, y_dev, y_test = y_train.T, y_dev.T, y_test.T

vocab_size = len(vocab_processor.vocabulary_)
embedding_size = 64
num_filters = 32
filter_sizes = [3, 4, 5]
hidden_units = 100
num_epochs = 100
mini_batch_size = 64
alpha = 0.009
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
keep_probs = [0.5, 0.5]

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

last_run = pickle.load(open("../runs/last_run.txt", "rb"))
pickle.dump(last_run + 1, open("../runs/last_run.txt", "wb"))

dump_dir = os.path.join("../runs", str(last_run))
os.makedirs(dump_dir)

load_params_file = None
# load_params_dir = "../runs/55"
# load_params_file = os.path.join(load_params_dir, "training_73.txt")
# hyperparams = pickle.load(open(os.path.join(load_params_dir, "hyperparams.txt"), "rb"))
# vocab_size = hyperparams["vocab_size"]
# embedding_size = hyperparams["embedding_size"]
# num_filters = hyperparams["num_filters"]
# filter_sizes = hyperparams["filter_sizes"]
# hidden_units = hyperparams["hidden_units"]
# num_epochs = hyperparams["num_epochs"]
# mini_batch_size = hyperparams["mini_batch_size"]
# alpha = hyperparams["alpha"]
# beta1 = hyperparams["beta1"]
# beta2 = hyperparams["beta2"]
# epsilon = hyperparams["epsilon"]
# keep_probs = hyperparams["keep_probs"]

# override some of the hyperparams
# keep_probs = [0.5, 0.5]
# alpha = 0.009

params = cnn.cnn(x_train, y_train, x_dev, y_dev, load_params_file, dump_dir, vocab_size, embedding_size,
                 num_filters, filter_sizes, hidden_units, num_epochs, mini_batch_size, alpha, beta1, beta2,
                 epsilon, keep_probs)
