
import pickle
import numpy as np

import sys
import os

import matplotlib.pyplot as plt
from cleverhans.utils_mnist import data_mnist

def vote_stats(scores):
    votes = np.argmax(scores, axis=1)
    vs = [0] * 10
    for v in votes:
        vs[v] += 1
    return vs


def visualize(image, stats, figure=None):
    # To avoid creating figures per input sample, reuse the sample plot
    if figure is None:
        plt.ion()
        #plt.ioff()
        figure = plt.figure()
        figure.canvas.set_window_title('Visualization')

        # magnify the image
        F = plt.gcf()
        sz = F.get_size_inches()
        F.set_size_inches( sz * 2.0 ) 


    else:
        figure.clf()

    ax = figure.add_subplot(1,2,1)

    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    elif len(image.shape) == 3:
        ax.imshow(image)
    else:
        print("Error: unexpected image shape: ", image.shape)

    plt.pause(0.1)

    #colors = ['green','blue','yellow','purple','red']
    #['NoAttack', 'jsma', 'deepfool', 'cw', 'fgsm']
    colors = {'NoAttack': 'green',
              'jsma' : 'blue',
              'deepfool' : 'yellow',
              'cw' : 'purple',
              'fgsm' : 'red'
            }

    ax = figure.add_subplot(1,2,2)
    xs = np.arange(10)
#    W = 0.8 / len(stats)
#    for idx,s in enumerate(stats):
#        ax.bar(xs + idx*W - 0.4, s, width=W, color=colors[idx], align='center')

    B = np.zeros(10)
    for key in stats:
        s2 = np.array( stats[key] )/1002.
        ax.bar(xs, s2, color=colors[key], align='center', bottom=B, label=key)
        B += s2

    #n_bins = 10
    #ax.hist(data, n_bins, histtype='bar', stacked=True)

    plt.xticks(xs, [ str(i) for i in range(10)])
    plt.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0.)

    x1,x2,y1,y2 = plt.axis()
    ylength = y2 - y1
    xlength = x2 - x1
    aspect_ratio = 1.0
    aspect = float(xlength)/ylength * aspect_ratio
    ax.set_aspect(aspect)


    plt.pause(0.1)
    plt.show()
    return figure


#pfile = 'pickle_dir/test_0_NoAttack.pkl'
#pfile = 'pickle_dir/test_0_deepfool.pkl'
#scores = pickle.load( open(pfile, 'rb') )
#print( vote_stats( scores ) ) 

# Get MNIST test data
X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                              train_end=0,
                                              test_start=0,
                                              test_end=500)

img_rows = 28
img_cols = 28

#sample = np.reshape( X_test[0], (img_rows, img_cols) )
#vs = vote_stats(scores)
#visualize(sample, [vs,vs,vs,vs])
#exit(0)

if len(sys.argv) != 2:
    print("Usage: %s pickle_dir" % sys.argv[0] )
    exit(0)

pickle_dir = sys.argv[1]


test_model_stats = {}
model_test_stats = {}

dbg_id = 0

for x in os.listdir( pickle_dir ):
    _, test_id, model = x.split('_')

    test_id = int(test_id)
    #if test_id != dbg_id:
    #    continue

    model = model.split('.')[0]
    pfile = os.path.join(pickle_dir, x)

    if not os.path.isfile(pfile):
        continue

    
    scores = pickle.load( open(pfile, 'rb') )
    vs = vote_stats( scores )

    if test_id not in test_model_stats:
        test_model_stats[ test_id ] = {}

    if model not in model_test_stats:
        model_test_stats[ model ] = {}

    test_model_stats[ test_id ][ model ] = vs
    model_test_stats[ model ][ test_id ] = vs 

#ms = ['NoAttack', 'jsma', 'deepfool', 'cw', 'fgsm']
#for m in ms:
#    if m in test_model_stats[dbg_id]:
#        print ("%s is found" % m)

#stats = [ test_model_stats[dbg_id][m] for m in ms ]
#print ("stats:", stats)

#print("test_model_stats:")
#print('len: ', len(test_model_stats))
#print("model_test_stats:")
#print('len: ', len(model_test_stats))

'''
figure = None
for key_id in test_model_stats:
    print("process id: %d" % key_id )
    dbg_id = key_id
    sample = np.reshape( X_test[dbg_id], (img_rows, img_cols) )
    figure = visualize( sample, test_model_stats[dbg_id], figure)
    figure.savefig('stats_imgs/test_%d.png' % dbg_id)
'''


