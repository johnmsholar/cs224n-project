import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import read_binaries
import itertools
import random

def generate_batch(article_input_matrix, headline_input_matrix, label_matrix):
  length = article_input_matrix.shape[0]
  indices = random.sample(range(length), BATCH_SIZE)
  article_batch = article_input_matrix[indices, :]
  headline_batch = headline_input_matrix[indices, :]
  label_batch = label_matrix[indices, :]
  return article_batch, headline_batch, label_batch

# Define Neural Net Parameters
# TODO: Change INPUT_DIM and NUM_CLASSES appropriately (currently set for training example)

INITIAL_EMBEDDING_DIM = 300
REDUCED_EMBEDDING_DIM = 100
BATCH_SIZE = 1000
INPUT_DIM = 200
NUM_CLASSES = 4
LEARNING_RATE = .01
NUM_ITERATIONS = 100

weight_names = ['W1', 'W2', 'W3']
bias_names = ['b1', 'b2', 'b3']

# Declare tf.placeholders for input data (X) and input labels (Y)
input_labels = tf.placeholder(tf.float32, (None, NUM_CLASSES))
article_input_matrix = tf.placeholder(tf.float32, (None, INITIAL_EMBEDDING_DIM))
headline_input_matrix = tf.placeholder(tf.float32, (None, INITIAL_EMBEDDING_DIM))

# Declare initial tf.Variable matrices to map from initial embeddings to reduced dimension embeddings
initial_article_weights = tf.Variable(tf.random_normal(shape=(INITIAL_EMBEDDING_DIM, REDUCED_EMBEDDING_DIM)))
initial_article_bias = tf.Variable(tf.zeros(shape=(1, REDUCED_EMBEDDING_DIM)))
initial_headline_weights = tf.Variable(tf.random_normal(shape=(INITIAL_EMBEDDING_DIM, REDUCED_EMBEDDING_DIM)))
initial_headline_bias = tf.Variable(tf.zeros(shape=(1, REDUCED_EMBEDDING_DIM)))

# Declare initial layer as a combination of article and headline initial layers
initial_article_layer = tf.nn.tanh(tf.matmul(article_input_matrix, initial_article_weights) + initial_article_bias)
initial_headline_layer = tf.nn.tanh(tf.matmul(headline_input_matrix, initial_headline_weights) + initial_headline_bias)
initial_layer = tf.concat(values=[initial_article_layer, initial_headline_layer], concat_dim=1)


# Declare tf.Variable weight matrices and bias vectors for 3 tanh layers
weights = [
  tf.Variable(tf.random_normal(shape=(INPUT_DIM, INPUT_DIM), stddev=1.0), name= n)
  for n in weight_names
]
biases = [
  tf.Variable(tf.zeros(shape=(1, INPUT_DIM)), name = n)
  for n in bias_names
]

# Declare tf.Variable weight matrix and bias vector for final layer
final_weights = tf.Variable(
  tf.random_normal(shape=(INPUT_DIM, NUM_CLASSES)),
  name='FinalWeights'
)
final_biases = tf.Variable(
  tf.zeros(shape=(1, NUM_CLASSES), dtype=tf.float32),
  name='FinalBiases'
)

# Declare intermediate variables as products of base variables for tanh layers and final layer
layer_1 = tf.nn.tanh(tf.matmul(initial_layer, weights[0]) + biases[0])
layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights[1]) + biases[1])
layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights[2]) + biases[2])
final_layer = tf.matmul(layer_3, final_weights) + final_biases

# Declare loss variable
loss = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=final_layer)
)

# Define train_step to be called repeatedly to optimize classification
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

"""
# Generate Sample Dataset in 2-dimensional space
# Class 1 is a Gaussian Centered Around (0, -5)
# Class 2 is a Gaussian Centered Around (0, 5)
# TODO: Remove this for final implementation

centroid_1 = np.array([0, -1])
centroid_2 = np.array([0, 1])
cov = np.array([
  [1, 0],
  [0, 1]
])
size = 500
x1, y1 = np.random.multivariate_normal(centroid_1, cov, size).T
x2, y2 = np.random.multivariate_normal(centroid_2, cov, size).T
labels_1 = np.concatenate([np.array([[1, 0]]) for _ in range(size)], axis=0)
labels_2 = np.concatenate([np.array([[0, 1]]) for _ in range(size)], axis=0)
x = np.concatenate([x1, x2], axis = 0).reshape((-1, 1))
y = np.concatenate([y1, y2], axis = 0).reshape((-1, 1))
all_data = np.concatenate([x, y], axis=1)
all_labels = np.concatenate([labels_1, labels_2], axis=0)

# Split example data into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
  all_data, all_labels, test_size=0.2, random_state=42
)
"""

# Run TF Session
# 1000 Train Iterations on Sample Data

sess = tf.Session()
sess.run(tf.global_variables_initializer())
X_train_input, X_test_input, y_train_input, y_test_input = read_binaries()
train_article_input_matrix, train_headline_input_matrix = X_train_input
test_article_input_matrix, test_headline_input_matrix = X_test_input

print('Binaries Initialized')
for iteration in range(NUM_ITERATIONS):
  article_batch, headline_batch, label_batch = generate_batch(
    train_article_input_matrix,
    train_headline_input_matrix,
    y_train_input
  )
  print('Iteration: ' + str(iteration))
  _ = sess.run(
    [train_step],
    feed_dict = {
      article_input_matrix : article_batch,
      headline_input_matrix : headline_batch,
      input_labels : label_batch
    }
  )
  loss_value = sess.run(
    [loss],
    feed_dict = {
      article_input_matrix : train_article_input_matrix,
      headline_input_matrix : train_headline_input_matrix,
      input_labels : y_train_input
    }
  )
  print('Loss: ' + str(loss_value))

# Evaluate performance on test set
print sess.run(
  loss, 
  feed_dict= { 
    article_input_matrix : test_article_input_matrix,
    headline_input_matrix : test_headline_input_matrix,
    input_labels : y_test_input
  }
)

# Plot data
"""
plt.plot(x1, y1, 'x')
plt.plot(x2, y2, 'x')
plt.axis('equal')
plt.savefig('plot.png', bbox_inches='tight')
"""
