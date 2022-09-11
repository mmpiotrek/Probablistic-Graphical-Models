import itertools
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.models import FactorGraph


def load_images():
    path_frames = "c6s1/frames/"
    images = []
    names = sorted(os.listdir(path_frames))
    for img_name in names:
        if img_name.endswith(".jpg"):
            img = cv2.imread(path_frames + img_name)
            images.append((img, img_name))
    return images


def load_data(img, img_name):
    path_bboxes = "c6s1/bboxes.txt"
    BBoxes = []
    with open(path_bboxes, "r") as f:
        lines = f.readlines()
        for line in lines:
            # find line with image name
            if img_name in line:
                # print(img_name)
                # # check next line, if next line is number (accept new line symbol), iterate over it and draw box
                if int(lines[lines.index(line) + 1].replace("\n", "")) > 0:
                    BB_qty = int(lines[lines.index(line) + 1].replace("\n", ""))
                    for i in range(BB_qty):
                        cords = lines[lines.index(line) + i + 2].split()
                        x = int(float(cords[0]))
                        y = int(float(cords[1]))
                        w = int(float(cords[2]))
                        h = int(float(cords[3]))
                        # reshape 10% each side
                        x = x + w//10
                        y = y + h//10
                        h = h * 8//10
                        w = w * 8//10
                        # print(x, y, w, h)
                        BBoxes.append(img[y:y + h, x:x + w])
    return BBoxes


def calculate_histograms(bbox_bgr):
    wh_ratio = bbox_bgr.shape[1]/bbox_bgr.shape[0]
    bbox_grey = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2GRAY)
    bbox_hsv = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(bbox_hsv)
    hist_grey = cv2.calcHist([bbox_grey], [0], None, [256], [0, 256])
    hist_hue = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_value = cv2.calcHist([v], [0], None, [256], [0, 256])
    histograms = [hist_grey, hist_hue, hist_saturation, hist_value]
    return histograms, wh_ratio


if __name__ == '__main__':
    images_bgr = load_images()
    previous_hists = -1
    previous_ratio = -1
    for img, name in images_bgr:
        current_hists = []
        current_ratio = []
        for i, box in enumerate(load_data(img, name)):
            BB_hist, ratio = calculate_histograms(box)
            current_hists.append(BB_hist)
            current_ratio.append(ratio)
        if len(current_hists) < 1:
            previous_hists = -1
            print()
            continue
        if previous_hists == -1:
            previous_hists = current_hists
            output = ""
            for i in range(len(current_hists)):
                output += "-1 "
            print(output)
            continue
        g = FactorGraph()
        fp_matrix = np.full((len(previous_hists) + 1, len(previous_hists) + 1), 1.0)
        for i in range(fp_matrix.shape[0]):
            for j in range(fp_matrix.shape[1]):
                if j == i:
                    if j != 0 and i != 0:
                        fp_matrix[i][j] = 0.0
        for i, current_hist in enumerate(current_hists):
            hist_avarage = []
            g.add_nodes_from([str(i)])
            for j, previous_hist in enumerate(previous_hists):
                compared_hist = []
                for itr in range(len(current_hist)):
                    comparation = 1.0 - cv2.compareHist(current_hist[itr], previous_hist[itr], cv2.HISTCMP_BHATTACHARYYA)
                    compared_hist.append(comparation)
                tmp_avarage = sum(compared_hist) / len(compared_hist)
                hist_avarage.append(tmp_avarage)
            print(hist_avarage)
            tmp = DiscreteFactor([str(i)], [len(previous_hists) + 1], [[0.65] + hist_avarage])
            g.add_factors(tmp)
            g.add_edge(str(i), tmp)

        if len(current_hists) > 1:
            tmp_list = list(range(len(current_hists)))
            for hist1, hist2 in itertools.combinations(tmp_list, 2):
                tmp = DiscreteFactor([str(tmp_list.index(hist1)), str(tmp_list.index(hist2))],
                                     [len(previous_hists) + 1, len(previous_hists) + 1], fp_matrix)
                g.add_factors(tmp)
                g.add_edge(str(tmp_list.index(hist1)), tmp)
                g.add_edge(str(tmp_list.index(hist2)), tmp)
        bp = BeliefPropagation(g)
        bp.calibrate()
        map_query = bp.map_query(g.get_variable_nodes(), show_progress=False)
        output = []
        for m in map_query:
            output.append(map_query[m] - 1)
        # print(output)
        print(' '.join(list(map(str, output))))
        previous_hists = current_hists
        previous_ratio = current_ratio
        if len(previous_hists) > 3:
            exit()