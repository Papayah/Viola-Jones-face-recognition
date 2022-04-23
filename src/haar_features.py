from src.integral_image import IntegralImage as Integral
from enum import Enum


def shift(pos: tuple, x: int, y: int):
    return (pos[0] + x, pos[1] + y)


class featureType(Enum):
    EDGE_VERTICAL = (1, 2)
    EDGE_HORIZONTAL = (2, 1)
    LINE_VERTICAL = (1, 3)
    LINE_HORIZONTAL = (3, 1)
    FOUR_RECTANGLE = (2, 2)


class HaarFeature(object):
    def __init__(self, type, pos, width, height, threshold, parity, weight=1):
        self.type: featureType = type
        self.top_left: tuple = pos
        self.width = width
        self.height = height
        self.bottom_right: tuple = (pos[0] + width, pos[1] + height)
        self.threshold = threshold
        self.parity = parity
        self.weight = weight

    def get_score(self, int_img: Integral):
        white, black = 0, 0

        if self.type == featureType.EDGE_VERTICAL:
            white += int_img.get_rect_sum(self.top_left, shift(self.bottom_right, 0, -self.height / 2))
            black += int_img.get_rect_sum(shift(self.top_left, 0, self.height / 2), self.bottom_right)

        elif self.type == featureType.EDGE_HORIZONTAL:
            white += int_img.get_rect_sum(self.top_left, shift(self.bottom_right, -self.width / 2, 0))
            black += int_img.get_rect_sum(shift(self.top_left, self.width / 2, 0), self.bottom_right)

        elif self.type == featureType.LINE_VERTICAL:
            white += int_img.get_rect_sum(self.top_left, shift(self.bottom_right, 0, -2 * self.height / 3))
            black += int_img.get_rect_sum(shift(self.top_left, 0, self.height / 3),
                                          shift(self.bottom_right, 0, -self.height / 3))
            white += int_img.get_rect_sum(shift(self.top_left, 0, 2 * self.height / 3), self.bottom_right)

        elif self.type == featureType.LINE_HORIZONTAL:
            white += int_img.get_rect_sum(self.top_left, shift(self.bottom_right, 0, -2 * self.width / 3))
            black += int_img.get_rect_sum(shift(self.top_left, self.width / 3, 0),
                                          shift(self.bottom_right, -self.width / 3, 0))
            white += int_img.get_rect_sum(shift(self.top_left, 2 * self.width / 3, 0), self.bottom_right)

        elif self.type == featureType.FOUR_RECTANGLE:
            white += int_img.get_rect_sum(self.top_left, shift(self.bottom_right, -self.width / 2, -self.height / 2))
            black += int_img.get_rect_sum(shift(self.top_left, self.width / 2, 0),
                                          shift(self.bottom_right, 0, -self.height / 2))
            white += int_img.get_rect_sum(shift(self.top_left, self.width / 2, self.height / 2), self.bottom_right)
            black += int_img.get_rect_sum(shift(self.top_left, 0, self.height / 2),
                                          shift(self.bottom_right, -self.width / 2, 0))

        return white - black

    def get_vote(self, int_img: Integral):
        score = self.get_score(int_img)
        return self.weight * (1 if score < self.parity * self.threshold else 0)
