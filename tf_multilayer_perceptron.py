import tensorflow as tf
import pandas as pd
import numpy as np
import helper_funcs as helpers
from mercedes_classes import DataFrameSelector, Dummifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score 

#Opening data
train_path = 'train.csv'

train = helpers.load_data(train_path, drop_cols=['ID'])
y_target = train["y"].values
train = train.drop(["y"], axis=1)

#Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(train, y_target, test_size=0.2, random_state=42)

#getting columns found in both data sets:
train_cols = helpers.get_cols(X_train)
test_cols = helpers.get_cols(X_test)
columns = [x for x in train_cols if x in test_cols]


#### Pipeline for processing data #######
pipeline = Pipeline([
        ('dummies', Dummifier()),
        ('selector', DataFrameSelector(columns)),
        ('std_scaler', StandardScaler()),
    ])

train_prepared = pipeline.fit_transform(X_train)
test_prepared = pipeline.transform(X_test)

# Parameters
learning_rate = 0.1
training_epochs = 50000
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 8 # 1st layer number of features
n_hidden_2 = 8 # 2nd layer number of features
n_input = 538 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.pow(pred-y, 2))/(2*train_prepared.shape[0])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

#Initializing generators
train_gen = group_list(train_prepared)
y_train_gen = group_list(y_train)

#Training 
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_prepared.shape[0]/batch_size)
        # Loop over all batches
        try:
            for i in range(total_batch):
                batch_x, batch_y = next(train_gen), next(y_train_gen).reshape(-1,1)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                    batch_y})
                # Compute average loss
                avg_cost += c / total_batch
        except StopIteration:
            pass
            # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    print(r2_score(y.eval(feed_dict={y: y_test.reshape(-1,1)}, session=sess), pred.eval(feed_dict={x: test_prepared}, session=sess)))
 
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_prepared, y: y_test.reshape(-1,1)}))


