import os

import cv2
import matplotlib.pyplot as plt


def compare(img_path1, img_path2):
    img1 = cv2.imread(img_path1, 0)
    img2 = cv2.imread(img_path2, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLAN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []

    for m1, m2 in matches:
        if m1.distance < 0.5 * m2.distance:
            good_matches.append([m1])
    flann_matches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
    plt.imshow(flann_matches)
    plt.show()
    return len(matches) > 5

path_to_imgs = 'feature_test'
for filename1 in os.listdir(path_to_imgs):
    for filename2 in os.listdir(path_to_imgs):
        if(filename1 == filename2): continue
        img1 = path_to_imgs + '\\' + filename1
        img2 = path_to_imgs + '\\' + filename2
        if compare(img1, img2):
            print(filename1, ' ', filename2, ' - same\n')

