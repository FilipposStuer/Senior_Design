import cv2
import numpy as np
import glob
list_of_images=[]
images_card=glob.glob("**/*ca*.jpg",recursive=True)
images_glass=glob.glob("**/*gl*.jpg",recursive=True)
images_metal=glob.glob("**/*me*.jpg",recursive=True)
images_paper=glob.glob("**/*pa*.jpg",recursive=True)
images_plastic=glob.glob("**/*pla*.jpg",recursive=True)
images_trash=glob.glob("**/*tra*.jpg",recursive=True)

for card in images_card:
    list_of_images.append(cv2.resize(cv2.imread(card,cv2.IMREAD_GRAYSCALE),(510,600)))
for glass in images_glass:
     list_of_images.append(cv2.resize(cv2.imread(glass,cv2.IMREAD_GRAYSCALE),(510,600)))
for metal in images_metal:
    list_of_images.append(cv2.resize(cv2.imread(metal,cv2.IMREAD_GRAYSCALE),(510,600)))
for paper in images_paper:
     list_of_images.append(cv2.resize(cv2.imread(paper,cv2.IMREAD_GRAYSCALE),(510,600)))
for plastic in images_plastic:
    list_of_images.append(cv2.resize(cv2.imread(plastic,cv2.IMREAD_GRAYSCALE),(510,600)))
for trash in images_trash:
     list_of_images.append(cv2.resize(cv2.imread(trash,cv2.IMREAD_GRAYSCALE),(510,600)))
label1=np.full(int(len(images_card)), 0,dtype=np.int32)
label2 =np.full(int(len(images_glass)), 1,dtype=np.int32)
label3=np.full(int(len(images_metal)), 2,dtype=np.int32)
label4 =np.full(int(len(images_paper)), 3,dtype=np.int32)
label5=np.full(int(len(images_plastic)), 4,dtype=np.int32)
label6 =np.full(int(len(images_trash)), 5,dtype=np.int32)
labels=np.concatenate((label1,label2,label3,label4,label5,label6))
array_of_images=np.array(list_of_images,dtype=np.float32)
print(len(list_of_images))
print(len(labels))
train_data = cv2.ml.TrainData_create(array_of_images.flatten(),cv2.ml.COL_SAMPLE,labels )
model=cv2.ml.DTrees_create()
model.setMaxDepth(5)
#model.setMinSampleCount(20)
#model.setUseSurrogates(False)
model.setCVFolds(0)
model.train(train_data)
sample=cv2.resize(cv2.imread("TrashType_Image_Dataset/cardboard/cardboard_001.jpg",cv2.IMREAD_GRAYSCALE).flatten(),(510,600)).astype(np.float32)
#print(model.predict(sample))



