import time

import cv2 as cv, cv2
import numpy as np
import os
from matplotlib import pyplot as plt

BBoxes = list()
amount_of_BB = range(0)
histograms = list()


class LabClass:
    img_color_list = []
    img_grey_list = []

    def __init__(self):

        self.load_images()

    def load_images(self):
        tmp = sorted(os.listdir("c6s1/frames"))
        for file in tmp:
            if file.endswith(".jpg"):
                img_color = cv.imread("c6s1/frames/" + file)
                # img_grey = cv.imread("c6s1/frames/" + file, cv.IMREAD_GRAYSCALE)
                self.img_color_list.append((file, img_color))
                # self.img_grey_list.append((file, img_grey))

    @staticmethod
    def resize(scale_percent: int = 100, image=None):
        """

        :param scale_percent:
        :param image:

        :return: resized image as np.ndarray
         """
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    #@staticmethod

    def drawBB(self, tup_img, img_num):
        """
        Draws bounding box on image
        :param img_num:
        :param tup_img:
        :return:
        """
        img = tup_img[1]
        img_name = tup_img[0]
        img_for_BB = tup_img[1]

        # read box cooringdinates from file
        with open(f"c6s1/bboxes.txt", "r") as f:

            lines = f.readlines()
            for line in lines:
                # find line with image name
                if img_name in line:
                    # check next line, if next line is number (accept new line symbol), iterate over it and draw box
                    # if len(lines[lines.index(line)+1])<4:

                    if lines[lines.index(line) + 1].replace("\n", "").isdigit() and img_num != len(BBoxes)-1:
                        current_BB = list()
                        current_histograms = list()
                        for i in range(int(lines[lines.index(line) + 1])):
                            global amount_of_BB
                            amount_of_BB = range(int(lines[lines.index(line) + 1]))
                            # get coordinates
                            coords = lines[lines.index(line) + i + 2].split()

                            BB = img_for_BB[int(float(coords[1])):int(float(coords[1])) + int(float(coords[3])),
                                 int(float(coords[0])):int(float(coords[0])) + int(float(coords[2]))]
                            BB = cv.cvtColor(BB, cv.COLOR_RGB2GRAY)

                            current_BB.append(BB)

                            # draw box
                            # cv.rectangle(img, (int(float(coords[0])), int(float(coords[1]))), (
                            #     int(float(coords[0])) + int(float(coords[2])),
                            #     int(float(coords[1])) + int(float(coords[3]))), (0, 255, 0), 2)
                            hist = self.plot_histogram(BB, img_num, i, True)
                            current_histograms.append(hist)
                        BBoxes.append((img_num, amount_of_BB, current_BB, current_histograms))
                        self.compare_BBoxes()
        return img

    def plot_histogram(self, src, num_img, num_BB, grey):
        if grey:
            bgr_planes = cv.split(src)
            histSize = 256
            histRange = (0, 255)  # the upper boundary is exclusive
            accumulate = True
            b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
            hist_w = 256
            hist_h = 200
            bin_w = int(round(hist_w / histSize))
            histImage = np.zeros((hist_h, hist_w, 1), dtype=np.uint8)
            cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
            for i in range(1, histSize):
                cv.line(histImage, (bin_w * (i - 1), hist_h - int(b_hist[i - 1])),
                        (bin_w * (i), hist_h - int(b_hist[i])),
                        (255, 0, 0), thickness=2)
            cv.imshow(f'hist from image: {num_img} for BB: {num_BB}', histImage)
            return b_hist

        else:
            bgr_planes = cv.split(src)
            histSize = 256
            histRange = (0, 255)  # the upper boundary is exclusive
            accumulate = True
            b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
            g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
            r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
            hist_w = 512
            hist_h = 400
            bin_w = int(round(hist_w / histSize))
            histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
            cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
            cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
            cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
            for i in range(1, histSize):
                cv.line(histImage, (bin_w * (i - 1), hist_h - int(b_hist[i - 1])),
                        (bin_w * (i), hist_h - int(b_hist[i])),
                        (255, 0, 0), thickness=2)
                cv.line(histImage, (bin_w * (i - 1), hist_h - int(g_hist[i - 1])),
                        (bin_w * (i), hist_h - int(g_hist[i])),
                        (0, 255, 0), thickness=2)
                cv.line(histImage, (bin_w * (i - 1), hist_h - int(r_hist[i - 1])),
                        (bin_w * (i), hist_h - int(r_hist[i])),
                        (0, 0, 255), thickness=2)
            cv.imshow(f'histogram: {num_img}, {num_BB}', histImage)
            return b_hist
        # histograms.append(b_hist)

    def compare_BBoxes(self):

        if len(BBoxes) >= 2:
            for i in range(len(BBoxes)):  # for i in range(len(BBoxes)-(len(BBoxes)-2)):
                img_num = BBoxes[i][0]
                BB_num = len(BBoxes[i][1])
                print(f'numer jpg: {img_num}, ilosc BB: {BB_num}')
                for j in range(len(BBoxes[i][2])):
                    cv.imshow(f'Image: {img_num}, nr BB: {j}', BBoxes[i][2][j])
                    print(f'aktualny BB: {j} \n ------------------')
                    for k in range(len(BBoxes[i-1][2])):
                        for compare_method in range(1):  # było 4
                            # j = len(BBoxes[i-1][3][j])
                            base_base = cv.compareHist(BBoxes[i][3][j], BBoxes[i-1][3][k], compare_method)
                            # base_half = cv.compareHist(BBoxes[i][3][j], BBoxes[i-1][3][k], compare_method)
                            # base_test1 = cv.compareHist(BBoxes[i][3][j], BBoxes[i-1][3][k], compare_method)
                            # base_test2 = cv.compareHist(BBoxes[i][3][j], BBoxes[i-1][3][k], compare_method)
                            print('Method:', compare_method, 'Perfect, Base-Half, Base-Test(1), Base-Test(2) :',
                                  base_base) #, '/', base_half, '/', base_test1, '/', base_test2)
                            # print('-------------------------')
                #cv.imshow(f'Hist {img_num}, {BB_num}', BBoxes[i][3][0])
        else:
            pass
            # for i in range(5):
            #     img_num = BBoxes[i][0]
            #     BB_num = BBoxes[i][1][-1]
            #     print(img_num, BB_num)
            #     for j in range(len(BBoxes[-i][2])):
            #         cv.imshow(f'BB {img_num}, {j}', BBoxes[-i][2][-j])
            #         print(j)
            #     pass # dodać tutaj dla 5 ostatnich

        # for i in amount_of_BB:
        #     BB = BBoxes[-i - 1]
        #     BB_grey = cv.cvtColor(BB, cv.COLOR_RGB2GRAY)
        #     BB_HSV = cv.cvtColor(BB, cv.COLOR_RGB2HSV)
        #     # blur = cv.blur(BB_grey, (6, 6))
        #     cv.imshow(f'numer zdjecia: {current_number}, numer BB: {i}', BB_grey)
        #     self.plot_histogram(BB_grey, current_number, i, True)
        #     if len(histograms) >= 5:
        #         for compare_method in range(4):
        #             base_base = cv.compareHist(histograms[-1], histograms[-2], compare_method)
        #             base_half = cv.compareHist(histograms[-1], histograms[-3], compare_method)
        #             base_test1 = cv.compareHist(histograms[-1], histograms[-4], compare_method)
        #             base_test2 = cv.compareHist(histograms[-1], histograms[-5], compare_method)
        #             print('Method:', compare_method, 'Perfect, Base-Half, Base-Test(1), Base-Test(2) :',
        #                   base_base, '/', base_half, '/', base_test1, '/', base_test2)  # Uwaga bo cały czas napierdala



    def show_images(self):
        cv.namedWindow('Image')
        # cv.createTrackbar('Img_sel', 'Image', 0, len(self.img_color_list)-1, lambda x: x)
        image_num = 0
        while True:
            # img_grey = self.img_grey_list[image_num]
            img_color = self.img_color_list[image_num]
            img_to_display = self.drawBB(img_color, image_num)
            cv.imshow('Image', self.resize(image=img_to_display))
            # image_num = cv.getTrackbarPos('Img_sel', 'Image')
            # self.compare_BBoxes(image_num)
            key_code = cv.waitKey(10)
            if key_code == 27:
                break
            elif key_code == ord('s'):
                image_num += 1
            elif key_code == ord('a'):
                image_num -= 1

            # print(img_color[0])


        cv.destroyAllWindows()
        return


if __name__ == "__main__":
    win = LabClass()
    win.show_images()
