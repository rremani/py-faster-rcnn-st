import numpy as np

INCORRECT = -1
def sortInternalClasses(classes):
	newClasses = np.array([])
	singleClass = []
	# assuming each row in classes is of the shape : "w,x,y,z,class_label"
	for cl in np.unique(classes[:,0]):
		index = np.where(classes[:,0] == cl)[0]
		singleClass = classes[[x for x in index],:]
		singleClass = singleClass[singleClass[:,1].argsort()]
		classes = np.delete(classes, ([x for x in index]), axis=0)
		if(newClasses.size == 0):
			newClasses = singleClass
		else :
			newClasses = np.concatenate((newClasses,singleClass),axis= 0)
		singleClass = []

# def checkAccuracy(image_name, image_detections):
# 	f = open('./test_Annotations/'+image_name+".txt")
# 	annotations =  []
# 	for line in f:
# 		row = line.split(';')
# 		annotations.append(row)
# 	annotations = sortInternalClasses(np.array(annotations))
# 	image_detections = sortInternalClasses(image_detections)
# 	if(len(image_detections) > len(annotations)):
# 		return INCORRECT
# 	if()

l = [[1,2],[1,3],[1,6],[2,2],[3,5],[3,1],[4,3],[1,1]]
l = np.array(l)
sortInternalClasses(l)