import math
import numpy as np


class IntegralImage(object):
    def __init__(self, img):
        self.img = img
        self.img_sq = img * img
        self.shape = (img.shape[0] + 1, img.shape[1] + 1)
        self.int_img = np.zeros(self.shape)
        self.int_img_sq = np.zeros(self.shape)
        self.variance = 0
        self.calculate()
        self.calculate_sq()
        self.get_variance()

    def calculate(self):
        for i in range(1, self.shape[0]):
            for j in range(1, self.shape[1]):
                self.int_img[i][j] = self.img[i - 1][j - 1] + self.int_img[i - 1][j]\
                                     + self.int_img[i][j - 1] - self.int_img[i-1][j-1]
                # print(
                #     f'{self.int_img[i][j]} = {self.img[i - 1][j - 1]} + {self.int_img[i - 1][j]} + {self.int_img[i][j - 1]}')
                # print(f'\n{self.int_img}\n')

    def calculate_sq(self):
        for i in range(1, self.shape[0]):
            for j in range(1, self.shape[1]):
                self.int_img_sq[i][j] = self.img_sq[i - 1][j - 1] + self.int_img_sq[i - 1][j] + self.int_img_sq[i][j - 1]

    def get_variance(self):
        N = (self.shape[0] - 1) * (self.shape[1] - 1)
        m = self.int_img[-1][-1] / N
        self.variance = (self.int_img_sq[-1][-1] / N) - math.pow(m, 2)

    def get_rect_sum(self, top_left, bottom_right):
        top_left = (top_left[1], top_left[0])
        bottom_right = (bottom_right[1], bottom_right[0])
        if top_left == bottom_right:
            return self.int_img[top_left]
        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])

        return self.int_img[bottom_right] + self.int_img[top_left] - self.int_img[bottom_left] - self.int_img[top_right]


if __name__ == '__main__':
    test = IntegralImage(np.array([[1,2,2,4,1],
                                  [3,4,1,5,2],
                                  [2,3,3,2,4],
                                  [4,1,5,4,6],
                                  [6,3,2,1,3]]))
    print(str(test.int_img) + '\n')
    # print(test.int_img_sq)
    print(test.variance)