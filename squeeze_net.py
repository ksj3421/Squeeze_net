# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 00:42:20 2018

@author: KangSuejung
"""
  
import tensorflow as tf
# import matplotlib.pyplot as plt

 

tf.set_random_seed(777)  # reproducibility

ILSVRC2012 = input_data.read_data_sets("ILSVRC2012_data/", one_hot=True)
#URL of ILSVRC2012 dataset -->  http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads 

# hyper parameters
learning_rate = 0.04
training_epochs = 1000
batch_size = 512


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
  
    
    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, (None, 224, 224, 3))

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 224, 224, 3])
            self.Y = tf.placeholder(tf.float32, [None, 1000])
            print(np.shape(X_img))
            # Convolutional Layer #1
            X_img = tf.pad(X_img,[[2,2],[2,2],[2,2],[2,2]],"Constant")
            conv1 = tf.layers.conv2d(inputs=X_img, filters=96, kernel_size=[7, 7],strides=(2,2),
                                     padding="VALID", activation=tf.nn.relu)
            print(np.shape(conv1))
            
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3],strides=2,padding="VALID")
            print(np.shape(pool1))
            #fire_module #2
            conv2_1 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=[1, 1],activation=tf.nn.relu)
            print(np.shape(conv2_1)) 
            conv2_e1x1 = tf.layers.conv2d(inputs=conv2_1, filters=64, kernel_size=[1, 1], activation=tf.nn.relu,padding="SAME")
            print(np.shape(conv2_e1x1)) 
            conv2_e3x3 = tf.layers.conv2d(inputs=conv2_1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu,padding="SAME")
            print(np.shape(conv2_e3x3))
            conv2 = tf.concat([conv2_e1x1, conv2_e3x3], 3)
            print(np.shape(conv2))            
            #fire_module #3
            conv3_1 = tf.layers.conv2d(inputs=conv2, filters=16, kernel_size=[1, 1],activation=tf.nn.relu)
            conv3_e1x1 = tf.layers.conv2d(inputs=conv3_1, filters=64, kernel_size=[1, 1], activation=tf.nn.relu,padding="SAME")
            conv3_e3x3 = tf.layers.conv2d(inputs=conv3_1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu,padding="SAME")
            conv3 = tf.concat([conv3_e1x1, conv3_e3x3], 3)
            
            #fire_module #4
            conv4_1 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[1, 1],activation=tf.nn.relu)
            conv4_e1x1 = tf.layers.conv2d(inputs=conv4_1, filters=128, kernel_size=[1, 1], activation=tf.nn.relu,padding="SAME")
            conv4_e3x3 = tf.layers.conv2d(inputs=conv4_1, filters=128, kernel_size=[3, 3], activation=tf.nn.relu,padding="SAME")
            conv4 = tf.concat([conv4_e1x1, conv4_e3x3], 3)
            
            # Pooling Layer #2
            pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3],strides=2,padding="VALID")
            print("pooling2 :",np.shape(pool2))
            #fire_module #5
            conv5_1 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[1, 1], activation=tf.nn.relu)
            conv5_e1x1 = tf.layers.conv2d(inputs=conv5_1, filters=128, kernel_size=[1, 1], activation=tf.nn.relu,padding="SAME")
            conv5_e3x3 = tf.layers.conv2d(inputs=conv5_1, filters=128, kernel_size=[3, 3], activation=tf.nn.relu,padding="SAME")
            conv5 = tf.concat([conv5_e1x1, conv5_e3x3], 3)
            print("fire 5 :",np.shape(conv5))
            #fire_module #6
            conv6_1 = tf.layers.conv2d(inputs=conv5, filters=48, kernel_size=[1, 1], activation=tf.nn.relu)
            conv6_e1x1 = tf.layers.conv2d(inputs=conv6_1, filters=192, kernel_size=[1, 1], activation=tf.nn.relu,padding="SAME")
            conv6_e3x3 = tf.layers.conv2d(inputs=conv6_1, filters=192, kernel_size=[3, 3], activation=tf.nn.relu,padding="SAME")
            conv6 = tf.concat([conv6_e1x1, conv6_e3x3], 3)
            print("fire 6 :",np.shape(conv6))
            #fire_module #7
            conv7_1 = tf.layers.conv2d(inputs=conv6, filters=48, kernel_size=[1, 1],activation=tf.nn.relu)
            conv7_e1x1 = tf.layers.conv2d(inputs=conv7_1, filters=192, kernel_size=[1, 1], activation=tf.nn.relu,padding="SAME")
            conv7_e3x3 = tf.layers.conv2d(inputs=conv7_1, filters=192, kernel_size=[3, 3], activation=tf.nn.relu,padding="SAME")
            conv7 = tf.concat([conv7_e1x1, conv7_e3x3], 3)
            print("fire 7 :",np.shape(conv7))
            #fire_module #8
            conv8_1 = tf.layers.conv2d(inputs=conv7, filters=64, kernel_size=[1, 1], activation=tf.nn.relu)
            conv8_e1x1 = tf.layers.conv2d(inputs=conv8_1, filters=256, kernel_size=[1, 1], activation=tf.nn.relu,padding="SAME")
            conv8_e3x3 = tf.layers.conv2d(inputs=conv8_1, filters=256, kernel_size=[3, 3], activation=tf.nn.relu,padding="SAME")
            conv8 = tf.concat([conv8_e1x1, conv8_e3x3], 3)
            print("fire 8 :",np.shape(conv8))
            # Pooling Layer #2
            pool3 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[3, 3],strides=2,padding="VALID")
            print("pooling3 :",np.shape(pool3))
            #fire_module #9
            conv9_1 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[1, 1],activation=tf.nn.relu)
            conv9_e1x1 = tf.layers.conv2d(inputs=conv9_1, filters=256, kernel_size=[1, 1], activation=tf.nn.relu,padding="SAME")
            conv9_e3x3 = tf.layers.conv2d(inputs=conv9_1, filters=256, kernel_size=[3, 3], activation=tf.nn.relu,padding="SAME")
            conv9 = tf.concat([conv9_e1x1, conv9_e3x3], 3)
            print("fire9 :",np.shape(conv9))
            dropout = tf.layers.dropout(inputs=conv9,rate=0.5, training=self.training)
           # Convolutional Layer #10
            net = tf.layers.conv2d(inputs=dropout, filters=1000, kernel_size=[1,1],activation=None)
            print("conv10 :",np.shape(net))
            h = tf.nn.avg_pool(net, ksize=[1, 13, 13, 1000], strides=[1, 1, 1, 1], padding='VALID')
            avgpool10 = tf.reduce_mean(h, axis=[1, 2])
            #avgpool10 = tf.contrib.layers.avg_pool2d(net, kernel_size =[13,13],stride=1,padding='VALID')
            print("avgpool:",np.shape(avgpool10))
            


            self.logits = tf.layers.dense(inputs=avgpool10 , units=1000)
            # define cost/loss & optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(self.cost)
    
            correct_prediction = tf.equal(
                tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})



# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(ILSVRC2012.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = ILSVRC2012.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', m1.get_accuracy(ILSVRC2012.test.images, ILSVRC2012.test.labels))
 
