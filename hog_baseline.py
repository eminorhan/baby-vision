'''HoG baseline'''
import os
import argparse
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Linear decoding with HoG model')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--subsample', default=False, action='store_true', help='subsample data?')

if __name__ == '__main__':
    args = parser.parse_args()

    c_list = os.listdir(args.data)
    c_list.sort()
    print('Class list:', c_list)

    imgs = []
    labels = []
    label_counter = 0
    file_counter = 0

    for c in c_list:
            curr_dir = os.path.join(args.data, c)
            f_list = os.listdir(curr_dir)
            f_list.sort()

            print('Reading class:', c)

            for f in f_list:
                    f_path = os.path.join(curr_dir, f)
                    img = imread(f_path)
                    feats = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(3, 3), block_norm='L2',
                                visualize=False, transform_sqrt=False, feature_vector=True, multichannel=True)
                    if args.subsample:
                            if file_counter % 10 == 0:
                                    imgs.append(feats)
                                    labels.append(label_counter)
                    else:
                            imgs.append(feats)
                            labels.append(label_counter)

                    file_counter += 1

            label_counter += 1

    imgs = np.vstack(imgs)
    labels = np.array(labels)

    print('Imgs shape:', imgs.shape)
    print('Labels shape:', labels.shape)

    print('Splitting dataset')
    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.5)

    print('Fitting training data')
    clf = SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001, max_iter=250)
    clf.fit(X_train, y_train)

    print('Computing predictions')
    pred_test = clf.predict(X_test)
    test_acc = np.mean(y_test==pred_test)

    pred_train = clf.predict(X_train)
    train_acc = np.mean(y_train==pred_train)

    print('Test accuracy', test_acc)
    print('Train accuracy', train_acc)