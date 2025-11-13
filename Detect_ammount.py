import cv2
import numpy as np



image = cv2.imread('balls.jpg',cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(image, (11, 11), 0)
canny = cv2.Canny(blur, 60 , 190, 3)
dilated = cv2.dilate(canny, (1, 1), iterations=0)



(objects, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
objectIm=cv2.drawContours(rgb, objects[0], -1, (0, 255, 0), 2)
cv2.imshow('Detected Objects', objectIm)
mask = np.zeros_like(image)


extracted_region = cv2.bitwise_and(objectIm, objectIm, mask=mask)
new_image = np.zeros_like(image)
cv2.imshow('Contoured', extracted_region)
print("objects in image : ", len(objects))

print(objects[0])


# _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)


# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# num_objects = len(contours)

# print(f"Number of objects detected: {num_objects}")

# # Optionally, draw contours for visualization
# output_image = image.copy()
# cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2) # Draw all contours in green

# cv2.imshow('Original Image', image)
# cv2.imshow('Detected Objects', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()