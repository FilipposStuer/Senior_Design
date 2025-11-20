import cv2
import numpy as np



image = cv2.resize(cv2.imread('balls3.jpg'),(510,600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11, 11), 0)
canny = cv2.Canny(blur, 0  , 130 , 3)
dilated = cv2.dilate(canny, (1, 1), iterations=0)



(objects, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
objectIm=cv2.drawContours(rgb, objects[0], -1, (0, 255, 0), 2)
cv2.imshow('Detected Objects', objectIm)
mask = np.zeros_like(gray)

cv2.drawContours(mask, objects, -1, (255,255,255), cv2.FILLED)
extracted_region = cv2.bitwise_and(image, image, mask=mask)
new_image = np.zeros_like(image)
cv2.imshow('Contoured', extracted_region)





mask2 = np.zeros_like(gray)
cv2.drawContours(mask2, objects[4:5], -1, (255,255,255), cv2.FILLED)
extracted_region2 = cv2.bitwise_and(image, image, mask=mask2)
cv2.imshow('Contoured2', extracted_region2)
print(extracted_region2)



print("objects in image : ", len(objects))
i=0
while i < len(objects):
    mask2 = np.zeros_like(gray)
    cv2.drawContours(mask2, objects[i:i+1], -1, (255,255,255), cv2.FILLED)
    extracted_region2 = cv2.bitwise_and(image, image, mask=mask2)
    file_name=f"imageSingle{i}.jpg"
    cv2.imwrite(file_name, extracted_region2)
    i+=1


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