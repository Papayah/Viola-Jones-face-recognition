import numpy as np
import pickle


from src.integral_image import IntegralImage as Integral
from src.haar_features import HaarFeature as Haar, HaarFeature
from src.haar_features import FeatureType
import cv2 as cv
from matplotlib import pyplot as plt

class HaarContainer(list):
    def __str__(self):
        for feat in self:
            print(str(feat))

    def __eq__(self, other):
        if(isinstance(self, HaarContainer)):
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


    img = cv.imread('face1.jpeg')
    #
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    #
    scale_percent = 10  # percent of original size
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

    for feat in haar_container:
        curr_score = feat.get_vote(integral_img)
        # print(curr_score)
        if curr_score == 0:
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