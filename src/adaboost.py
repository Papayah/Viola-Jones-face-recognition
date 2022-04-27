import numpy as np

from src.integral_image import IntegralImage as Integral
from src.haar_features import HaarFeature as Haar
from src.haar_features import FeatureType
import cv2 as cv
from matplotlib import pyplot as plt


def create_features(img_width, img_height, min_feat_width, max_feat_wid, min_feat_height, max_feat_height):
    # private function to create all possible features
    # return haar_feats: a list of haar-like features
    # return type: np.array(haar.HaarLikeFeature)

    haar_feats = list()

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
                        haar_feats.append(Haar(feat_type, (i, j), feat_width, feat_height, 0,
                                               1))  # threshold = 0 (no misclassified images)
                        haar_feats.append(Haar(feat_type, (i, j), feat_width, feat_height, 0,
                                               -1))  # threshold = 0 (no misclassified images)

    return haar_feats










img= cv.imread('face1.jpeg')

img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

integral_img = Integral(img)

#print(create_features(width, height, 1, width // 3, 1, height // 3))
for feat in create_features(20, 20, 1, 20 // 3, 1, 20 // 3):
    print(feat)
plt.imshow(img, cmap='gray')
plt.show()