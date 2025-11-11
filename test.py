import cv2 as cv
import numpy as np

    # Example: Creating dummy data
train_features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
train_labels = np.array([0, 1, 0], dtype=np.int32)

    # Create TrainData object
train_data = cv.ml.TrainData.create(train_features, cv.ml.ROW_SAMPLE, train_labels)
knn = cv.ml.KNearest_create()
knn.train(train_data)