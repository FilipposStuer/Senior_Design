import cv2
import numpy as np
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
    

frame = cv2.imread("marker23.png")
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)