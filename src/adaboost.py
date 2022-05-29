from functools import partial
import numpy as np
import os
import pickle

from src.integral_image import IntegralImage as Integral
from src.haar_features import HaarFeature as Haar
from src.haar_features import FeatureType
import cv2 as cv
from matplotlib import pyplot as plt

import progressbar
from multiprocessing import cpu_count, Pool

LOADING_BAR_LENGTH = 25


class HaarContainer(list):
    def __str__(self):
        for feat in self:
            print(str(feat))

    def __eq__(self, other):
        if isinstance(self, HaarContainer):
            if len(self) == len(other):
                for i in range(len(self)):
                    if self[i] != other[i]:
                        return False
                return True
            else:
                return False
        return False


def create_features(img_width, img_height, min_feat_width, max_feat_wid, min_feat_height, max_feat_height):
    # private function to create all possible features
    # return haar_feats: a list of haar-like features
    # return type: np.array(haar.HaarLikeFeature)

    haar_feats = list()
    counter = 0
    # iterate according to types of rectangle features
    for feat_type in FeatureType:

        # min of feature width is set in featureType enum
        feat_start_width = max(min_feat_width, feat_type.value[0])

        # iterate with step
        for feat_width in range(feat_start_width, max_feat_wid, feat_type.value[0]):

            # min of feature height is set in featureType enum
            feat_start_height = max(min_feat_height, feat_type.value[1])

            # iterate with setp
            for feat_height in range(feat_start_height, max_feat_height, feat_type.value[1]):

                # scan the whole image with sliding windows (both vertical & horizontal)
                for i in range(img_width - feat_width):
                    for j in range(img_height - feat_height):
                        haar1 = Haar(feat_type, (i, j), feat_width, feat_height, 0, 1)
                        haar2 = Haar(feat_type, (i, j), feat_width, feat_height, 0, -1)
                        haar_feats.append(haar1)
                        haar_feats.append(haar2)
                        counter += 1
                        if counter % 300 == 0:
                            print(counter)
    return HaarContainer(haar_feats)


def save_features(haar_container):
    with open(f'../haar_features_instances/all_features.pickle', 'wb') as file:
        pickle.dump(haar_container, file)


def load_features():
    with open(f'../haar_features_instances/all_features.pickle', 'rb') as file:
        haar_container = pickle.load(file)
    return haar_container


def get_feature_vote(feature, image):
    return feature.get_vote(image)


def save_votes(votes):
    np.savetxt("./data/votes.txt", votes, fmt='%f')
    print("...votes saved\n")


def load_votes():
    votes = np.loadtxt("./data/votes.txt", dtype=np.float64)
    return votes


def learn(pos_int_img, neg_int_img, num_classifiers=-1, min_feat_width=1, max_feat_width=-1, min_feat_height=1,
          max_feat_height=-1):

    num_pos = len(pos_int_img)
    num_neg = len(neg_int_img)
    num_imgs = num_pos + num_neg
    img_height, img_width = pos_int_img[0].shape

    max_feature_width = img_width if max_feat_width == -1 else max_feat_width
    max_feature_height = img_height if max_feat_height == -1 else max_feat_height

    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))

    images = pos_int_img + neg_int_img

    print("\ncreating haar-like features ...")
    features = create_features(img_width, img_height, min_feat_width, max_feature_width, min_feat_height,
                               max_feature_height)

    print('... done. %d features were created!' % len(features))

    num_features = len(features)
    feature_index = list(range(num_features))

    num_classifiers = num_features if num_classifiers == -1 else num_classifiers

    print("\ncalculating scores for images ...")

    if os.path.exists("./data/votes.txt"):
        votes = load_votes()
    else:
        votes = np.zeros((num_imgs, num_features))

        bar = progressbar.ProgressBar()

        NUM_PROCESS = cpu_count() * 3
        pool = Pool(processes=NUM_PROCESS)

        for i in bar(range(num_imgs)):
            votes[i, :] = np.array(list(pool.map(partial(get_feature_vote, image=images[i]), features)))

        save_votes(votes)

    classifiers = list()

    print("\nselecting classifiers ...")

    bar = progressbar.ProgressBar()

    for _ in bar(range(num_classifiers)):

        classification_errors = np.zeros(len(feature_index))

        weights *= 1. / np.sum(weights)

    #     for f in range(len(feature_index)):
    #         f_idx = feature_index[f]
    #         error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0,
    #                         range(num_imgs)))
    #         classification_errors[f] = error
    #
    #     min_error_idx = np.argmin(classification_errors)
    #     best_error = classification_errors[min_error_idx]
    #     best_feature_idx = feature_index[min_error_idx]
    #
    #     best_feature = features[best_feature_idx]
    #     feature_weight = .5 * np.log((1 - best_error) / best_error)  # alpha
    #     best_feature.weight = feature_weight
    #     classifiers.append(best_feature)
    #
    #     def new_weights(best_error):
    #         return np.sqrt((1 - best_error) / best_error)
    #
    #     weights_map = map(lambda img_idx: weights[img_idx] * new_weights(best_error) if labels[img_idx] != votes[
    #         img_idx, best_feature_idx] else weights[img_idx] * 1, range(num_imgs))
    #     weights = np.array(list(weights_map))
    #
    #     feature_index.remove(best_feature_idx)
    #
    # print("\nclassified selected ...\nreaching the end of AdaBoost algorithm ...")
    #
    # return classifiers

    #TEEEEEEEEEEEEEEEEEEEEEEEEEEEEEST


if __name__ == '__main__':
    # haar_container = create_features(50, 50, 1, 50 // 3, 1, 50 // 3)
    # save_features(haar_container)

    haar_container = load_features()

    # print(str(haar_container[0]))

    # print(haar_container1 == haar_container2)

    # for i in range(max(len(haar_container1), len(haar_container2))):
    #     feat1, feat2 = haar_container1[i], haar_container2[i]
    #     if feat1 != feat2:
    #         print(f'{feat1}\nNOT EQUAL\n{feat2}')

    img = cv.imread('face2.jpg')
    #
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    #
    scale_percent = 5  # percent of original size
    width = int(img.shape[1] * scale_percent // 100)
    height = int(img.shape[0] * scale_percent // 100)
    dim = (width, height)
    #
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    #
    integral_img = Integral(img)
    #
    # # print(create_features(width, height, 1, width // 3, 1, height // 3))
    max_feats = []
    max_score = 0

    for feat in haar_container:
        curr_score = feat.get_score(integral_img)
        curr_vote = feat.weight * (1 if curr_score < feat.parity * feat.threshold else 0)
        # print(curr_score)
        if abs(curr_score) > max_score and curr_vote == 0:
            max_score = curr_score
            max_feats.append(feat)
    # print(max_feats)
    # print(max_score)

    for max_feat in max_feats:
        for i in range(max_feat.width):
            img[i][max_feat.top_left[1]] = 0
            img[i][max_feat.bottom_right[1]] = 0

        for i in range(max_feat.height):
            img[max_feat.top_left[0]][i] = 0
            img[max_feat.bottom_right[0]][i] = 0

    plt.imshow(img, cmap='gray')
    plt.show()
