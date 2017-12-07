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

from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans_tutorials.tutorial_models import make_basic_cnn

FLAGS = flags.FLAGS

def visualize( stats, figure=None):

    # To avoid creating figures per input sample, reuse the sample plot
    if figure is None:
        #plt.ion()
        #plt.ioff()
        figure = plt.figure()
        figure.canvas.set_window_title('Visualization')


    K = len(stats)
    i = 0
    for image,scores in stats:
        i += 1
        ax = figure.add_subplot(1, K, i)
        #ax.axis('off')

        top_K = sorted(enumerate(scores), 
                key = lambda val : val[1], 
                reverse=True)

        display = [ "%d (%.6f%%)" % (x,y*100.0) for (x,y) in top_K ]

        ax.set_xlabel(",\n".join(display[:3]) )

        # If the image is 2D, then we have 1 color channel
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        elif len(image.shape) == 3:
            ax.imshow(image)
        else:
            print("Error: unexpected image shape: ", image.shape)


        # Give the plot some time to update
        plt.pause(0.01)

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
    viz_enabled = True

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

    sample = X_test[0]
    ss = [sample]
    noise = np.random.rand(img_rows,img_cols,1) 
    ss.append( sample + noise )
    ss.append( sample + (noise / 10.0) )
    ss.append( sample + (noise / 100.0) )


    figure = None
    with sess.as_default():
        K = len(ss)
        feed_dict = {x : ss}
        res = sess.run(tf.nn.softmax(preds), feed_dict=feed_dict)
        shows= []
        for i in range(K):
            img = np.reshape( ss[i], (img_rows, img_cols))
            shows.append( (img,res[i]) )
        figure = visualize(shows, figure)


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
