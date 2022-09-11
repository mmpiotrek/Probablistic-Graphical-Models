import itertools
import os
import cv2
import sys
import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.models import FactorGraph


# function to load frames from given directory
def load_images(path):
    path_frames = path + "/frames/"
    images = []
    names = sorted(os.listdir(path_frames))
    for img_name in names:
        if img_name.endswith(".jpg"):
            img = cv2.imread(path_frames + img_name)  # load only jpg format images in BGR
            images.append((img, img_name))
    return images


# function to extract bounding boxes
def load_data(path, img, img_name):
    path_bboxes = path + "/bboxes.txt"
    BBoxes = []
    with open(path_bboxes, "r") as f:
        lines = f.readlines()
        for line in lines:
            # find line with image name
            if img_name in line:
                # read quantity of bounding boxes
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
                        BBoxes.append(img[y:y + h, x:x + w])
    return BBoxes


# calculate factors to compare
def calculate_histograms(bbox_bgr):
    wh_ratio = bbox_bgr.shape[1]/bbox_bgr.shape[0]  # width/height ratio
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
    path_from_user = sys.argv[1]
    images_bgr = load_images(path_from_user)  # loading images
    previous_hists = -1  # initiation of values
    previous_ratio = -1
    for img, name in images_bgr:
        current_hists = []
        current_ratio = []
        for i, box in enumerate(load_data(path_from_user, img, name)):  # getting bounding boxes from image
            BB_hist, ratio = calculate_histograms(box)
            current_hists.append((i, BB_hist))  # adding BBox ID and BBox histograms to list
            current_ratio.append(float(ratio))  # adding w/h ratio to list
        if len(current_hists) < 1:  # if there isn't any BBox then print nothing and go to next loop iteration
            previous_hists = -1
            previous_ratio = current_ratio
            print()
            continue
        if previous_hists == -1:  # if it's first frame then print -1 for each BBox
            previous_hists = current_hists
            previous_ratio = current_ratio
            output = ""
            for i in range(len(current_hists)):
                output += "-1 "
            print(output)
            continue

        # create Factor Graph
        g = FactorGraph()

        # create matrix which is factor between nodes
        fp_matrix = np.full((len(previous_hists) + 1, len(previous_hists) + 1), 1.0)
        for i in range(fp_matrix.shape[0]):
            for j in range(fp_matrix.shape[1]):
                if j == i:
                    if j != 0 and i != 0:
                        fp_matrix[i][j] = 0.0  # set values to zero on diagonal (except for [0,0])

        # calculate similarity of bounding boxes
        for i, current_hist in enumerate(current_hists):
            hist_average = []
            g.add_nodes_from([str(i)])
            for j, previous_hist in enumerate(previous_hists):  # compare all previous histograms to current histograms
                compared_hist = []
                compared_ratio = current_ratio[i] / previous_ratio[j]
                for itr in range(len(current_hist[1])):  # compare all histograms from bounding boxes
                    comparison = 1.0 - cv2.compareHist(current_hist[1][itr], previous_hist[1][itr],
                                                       cv2.HISTCMP_BHATTACHARYYA)
                    compared_hist.append(comparison)
                # count average from histograms and weighted average with other factors
                tmp_average = sum(compared_hist) / len(compared_hist)
                score = 0.9 * tmp_average + 0.1 * compared_ratio
                hist_average.append(score)
            # print(hist_average)
            # set 0.65 as new person threshold and build graph
            tmp = DiscreteFactor([str(i)], [len(previous_hists) + 1], [0.65] + hist_average)
            g.add_factors(tmp)
            g.add_edge(str(i), tmp)

        # if there is more than one BBox build edges and factors to each node
        if len(current_hists) > 1:
            tmp_list = list(range(len(current_hists)))
            for hist1, hist2 in itertools.combinations(tmp_list, 2):
                tmp = DiscreteFactor([str(hist1), str(hist2)],
                                     [len(previous_hists) + 1, len(previous_hists) + 1], fp_matrix)
                g.add_factors(tmp)
                g.add_edge(str(hist1), tmp)
                g.add_edge(str(hist2), tmp)

        bp = BeliefPropagation(g)
        map_query = bp.map_query(g.get_variable_nodes(), show_progress=False)
        output = []
        # read and print dictionary in proper order using BBox ID
        for i in current_hists:
            output.append(map_query[str(i[0])] - 1)
        print(*output)

        # set current values to previous values at the end of the loop
        previous_hists = current_hists
        previous_ratio = current_ratio
