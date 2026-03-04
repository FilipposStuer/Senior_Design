#This file will use openCV to dtect the arm using ArUco
import cv2 
import numpy as np
markerImage = np.zeros((200, 200), dtype=np.uint8)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
cv2.aruco.generateImageMarker(dictionary, 23, 200, markerImage, 1)
cv2.imwrite("marker23.png", markerImage)