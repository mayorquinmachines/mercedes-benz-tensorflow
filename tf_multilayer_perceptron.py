import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score 
from helper_funcs import group_list
from data_prep import DataPrep

#Loading and Transforming data
prep = DataPrep(dummy_pipe=True)
X_train, y_train, X_val, y_val, test, test_id = prep.load_data('data/train.csv', 'data/test.csv')
X_train, X_val, _ = prep.transform(X_train, X_val, test)
#Don't need these for this example
del test
del test_id


# Parameters
learning_rate = 0.1
training_epochs = 50000
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 8 # 1st layer number of features
n_hidden_2 = 8 # 2nd layer number of features
n_input = X_train.shape[1] # MNIST data input (img shape: 28*28)
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

cost = tf.reduce_mean(tf.pow(pred-y, 2))/(2*X_train.shape[0])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()


#Training 
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        #Initializing generators
        train_gen = group_list(X_train, batch_size)
        y_train_gen = group_list(y_train, batch_size)
        # Loop over all batches
        try:
            for i in range(total_batch):
                batch_x, batch_y = next(train_gen), next(y_train_gen).reshape(-1,1)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y:batch_y})
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
    print(r2_score(y.eval(feed_dict={y: y_val.reshape(-1,1)}, session=sess), pred.eval(feed_dict={x: X_val}, session=sess)))
 
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_val, y: y_val.reshape(-1,1)}))


