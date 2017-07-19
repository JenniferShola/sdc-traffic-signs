from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
import numpy as np
import pickle
import math
import os

def shuffle(features, labels):
    indices = np.arange(0, len(features))
    np.random.shuffle(indices)
    return (features[indices], labels[indices])

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def normalize_grayscale(image_data):
    a = 0.1
    b = 0.9
    min_color = 0
    max_color = 255
    return a + (((image_data - min_color) * (b - a)) / (max_color - min_color))

def all_grey(train_features):
    features = []
    for f in train_features:
        features.append(normalize_grayscale(grayscale(f)))
    return np.array(features)

# Import training and test data
training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train, z_size, c_train = train['features'], train['labels'], train['sizes'], train['coords']
X_test, y_test = test['features'], test['labels']
X_train, y_train = shuffle(X_train, y_train)

# To start off let's do a basic data summary.
n_train = X_train.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1]
n_classes = 43

# TODO: Adding label names to make feature debugging easier
label_names = {}
with open("signnames.csv", mode='r') as f:
    for line in f:
        splits = line.strip().split(',')
        if splits[0] == 'ClassId':
            continue
        label_names[int(splits[0])] = splits[1]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("")

for label in range(0,43):
    print("Label Name for #" + str(label) + " is " + label_names[label])

# Parameters
learning_rate = 0.001
batch_size = 128
#batch_size = 512
training_epochs = 2
validation_size=5000

n_input = 1024  # grayscale German Sign data input (Shape: 32*32 * 1)
n_classes = 43  # total classes

layer_width = {
    'layer_1': 32,
    'layer_2': 64,
    'layer_3': 128,
    'layer_4': 256,
    'fully_connected': 512
}

# Store layers weight & bias
weights = {
    'layer_1': tf.Variable(tf.truncated_normal([5, 5, 1, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal([5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'layer_3': tf.Variable(tf.truncated_normal([5, 5, layer_width['layer_2'], layer_width['layer_3']])),
    'layer_4': tf.Variable(tf.truncated_normal([5, 5, layer_width['layer_3'], layer_width['layer_4']])),
    'fully_connected': tf.Variable(tf.truncated_normal([1024, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal([layer_width['fully_connected'], n_classes]))
}

biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'layer_4': tf.Variable(tf.zeros(layer_width['layer_4'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(n_classes))
}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

def batches(index, total_batch, batch_size):
    start = index*batch_size
    end = (index+1)*batch_size
    if index < total_batch-1:
        new_features = train_features[start:end]
        new_labels = train_labels[start:end]
    else:
        new_features = train_features[start:]
        new_labels = train_labels[start:]
    return new_features, new_labels

def OHE_labels(train_ylabels, num_classes):
    OHC = OneHotEncoder()
    Y_ohc = OHC.fit(np.arange(num_classes).reshape(-1, 1))
    y_labels = Y_ohc.transform(train_ylabels.reshape(-1, 1)).toarray()
    return y_labels

# Create model
def conv_net(x, weights, biases):
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2)

    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv3 = maxpool2d(conv3)

    conv4 = conv2d(conv3, weights['layer_4'], biases['layer_4'])
    conv4 = maxpool2d(conv4)

    fc1 = tf.reshape(conv4, [-1, weights['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fully_connected']), biases['fully_connected'])
    fc1 = tf.nn.tanh(fc1)
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Main Code Starts Here
train_features = all_grey(X_train)
train_labels = OHE_labels(y_train, n_classes)

x = tf.placeholder("float", [None, 32, 32])
y = tf.placeholder("float", [None, n_classes])
x /= 255.
image = tf.reshape(x, [-1,32,32,1])
gray = tf.image.rgb_to_grayscale(image, name='grayscale')

logits = conv_net(gray, weights, biases)
    
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(n_train/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = batches(i, total_batch, batch_size) 
            
 	    # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        
	# Display logs per epoch step
        c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.20f}".format(c))
    
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    correct_prediction = tf.reshape(correct_prediction, [None,32,32])
 
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))






