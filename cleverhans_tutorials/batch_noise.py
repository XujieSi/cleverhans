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
import pickle

import matplotlib.pyplot as plt

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import NoAttack

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

        # magnify the image
        F = plt.gcf()
        sz = F.get_size_inches()
        F.set_size_inches( sz * 1.5 ) 

    else:
        # clear current figure
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

    figure = None

    for test_ind in range(0,2):

        sample2 = X_test[test_ind: test_ind + 1]
        current_class = int( np.argmax(Y_test[0]) )
        target = 3
        one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
        one_hot_target[0, target] = 1
        one_hot_target = None

        adv_models_params = []

        # try no attack
        adv_models_params.append( ('NoAttack', NoAttack(model, sess=sess),{}))


        # try JSMA 
        jsma = SaliencyMapMethod(model, sess=sess)
        jsma_params = {'theta': 1., 'gamma': 0.1,
                       'clip_min': 0., 'clip_max': 1.,
                       'y_target': one_hot_target}

        adv_models_params.append(  ('jsma', jsma, jsma_params)  )


        # try cw
        cw = CarliniWagnerL2(model, sess=sess)
        cw_params = {'binary_search_steps': 1,
                     'max_iterations': 100,
                     'learning_rate': 0.1,
                     'batch_size': 1,
                     'initial_const': 10,
                     'y_target': one_hot_target}
        adv_models_params.append(  ('cw', cw, cw_params)  )


        # try deepfool
        deepfool = DeepFool(model, sess=sess)
        df_params = {'max_iter': 20, 'nb_classes': 10,
                     'clip_min': 0., 'clip_max': 1.,
                     'overshoot': 0.1 }
        adv_models_params.append(  ('deepfool', deepfool, df_params)  )


        # try fast gradient sign
        fgsm = FastGradientMethod(model, sess=sess)
        fgsm_params = {'eps' : 0.3,
                       'clip_min': 0., 'clip_max': 1.,
                       'y_target': one_hot_target}
        adv_models_params.append(  ('fgsm', fgsm, fgsm_params)  )

        noise_candidates = []
        num_noise = 10
        for i in range(num_noise):
            # uniform distribution
            # noise = np.random.rand(img_rows,img_cols,1) / 2.0
            noise = np.random.randn(img_rows,img_cols,1) / 10.0
            noise_candidates.append( noise )

        for name, adv_model, params in adv_models_params:
            adv = adv_model.generate_np(sample2, **params)
            ae = adv[0]

            # visualize 
            sample = sample2[0]
            ss = [sample]
            desc = [ 'original' ]


            ss.append( ae )
            desc.append( name )

            for idx, noise in enumerate(noise_candidates):
                ss.append( ae + noise )
                desc.append( '%s_noise-%d' % (name,idx) )

            with sess.as_default():
                feed_dict = {x : ss}
                res = sess.run(tf.nn.softmax(preds), feed_dict=feed_dict)
                with open('pickle_dir/test_%d_%s.pkl'%(test_ind, name), 'wb') as fout:
                    pickle.dump(res,fout)

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
