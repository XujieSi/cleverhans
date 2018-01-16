from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
import os

import matplotlib.pyplot as plt

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import Caricature
from cleverhans.attacks import CaricatureFGSM
from cleverhans.attacks import CarliniWagnerL2
#from cleverhans.attacks import DeepFool
from cleverhans.attacks import FastGradientMethod

from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans_tutorials.tutorial_models import make_basic_cnn

FLAGS = flags.FLAGS

def visualize( stats, figure=None):

    # To avoid creating figures per input sample, reuse the sample plot
    if figure is None:
        plt.ion()
        #plt.ioff()
        figure = plt.figure()
        figure.canvas.set_window_title('Visualization')
    else:
        figure.clf()


    K = len(stats)
    Row,Col = 1,K
    Row,Col = int((K+1)/2) ,2


    i = 0
    for image,scores,desc in stats:
        i += 1
        ax = figure.add_subplot(Row, Col, i)
        ax.axis('off')

        top_K = sorted(enumerate(scores), 
                key = lambda val : val[1], 
                reverse=True)
        display = [ "%d (%.6f%%)" % (x,y*100.0) for (x,y) in top_K ]

        #ax.set_ylabel(",\n".join(display[:3]) )
        #ax.set_ylabel( desc )
        ax.text(30, 25, desc + '\n' + '\n'.join(display[:3]) )

        # If the image is 2D, then we have 1 color channel
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        elif len(image.shape) == 3:
            ax.imshow(image)
        else:
            print("Error: unexpected image shape: ", image.shape)


        # Give the plot some time to update
        #plt.pause(0.01)

    plt.pause(0.5)
    plt.show()
    return figure


def predict(model_path, image_path):
    """
    predict the label with given model and image
    :param model_path: path to a NN model
    :param image_path: path to an image
    """

    train_start = 0
    train_end = 60000
    test_start = 0
    test_end = 10000
    batch_size = 128

    source_sample = 10
    nb_classes = 10

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    model = make_basic_cnn()
    preds = model(x)

    print("Defined TensorFlow model graph.")
    rng = np.random.RandomState([2017, 8, 30])

    if os.path.exists(model_path + ".meta"):
        with sess.as_default():
            saver = tf.train.Saver()
            #saver = tf.train.Saver( allow_empty = True )
            saver.restore(sess, model_path)
    else:
        print("Error: cannot find the model (", model_path, ".meta)")
        sess.close()
        exit()

    #test_ind = 17 #14 #12
    figure = None

    sample_X = X_test[:400]
    sample_Y = Y_test[:400]
    ground_labels = np.argmax(sample_Y, axis=1)

    def label_shift(i,j):
        s_text = "shift("
        if i != 0:
            s_text += ('U' if i < 0 else 'D') + '%d' % (np.absolute(i))
        else:
            s_text += 'N'

        s_text += ','
        if j != 0:
            s_text += ('L' if j < 0 else 'R') + '%d' % (np.absolute(j))
        else:
            s_text += 'N'
        s_text += ')'

        return s_text


    M = 2
    shows = []
    for i in range(-M,M):
        for j in range(-M,M):
            t1 = np.roll(sample_X, i, axis=1)
            ss = np.roll(t1, j, axis=2)

            with sess.as_default():
                feed_dict = {x : ss}
                res = sess.run(preds, feed_dict=feed_dict)
                pred_labels = np.argmax(res, axis=1)
                for k in range(len(ss)):
                    if pred_labels[k] == ground_labels[k]:
                        continue

                    img0 = np.reshape( sample_X[k], (img_rows, img_cols) )
                    img1 = np.reshape( ss[k], (img_rows, img_cols) )

                    shows.append( (img0, sample_Y[k], "original(test%d)" % k) )
                    shows.append( (img1, res[k], label_shift(i,j) ) )
     
            print( "i=%d, j=%d" % (i,j), "shows size:", len(shows))

    print("shows size:", len(shows))
    figure = None
    i = 0
    while i < len(shows):
        e = min(i+10, len(shows) )
        figure = visualize( shows[i:e], figure)
        figure.savefig("shift_stats/test_%d_%d.jpg" % (i,e))
        i += 10

    exit()


    # close tf session
    sess.close()



def main(argv=None):
    predict(model_path = FLAGS.model_path,
            image_path = FLAGS.image_path)


if __name__ == '__main__':
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_string('image_path', os.path.join("imgs", "test0"),
                        'Path of the image to be predicted')

    tf.app.run()
