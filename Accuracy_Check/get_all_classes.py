f = open('./ImageSets/train.txt')
classes={}
for line in f:
	id_ = line.split('\n')[0]
	f1 = open("./Annotations/"+id_+'.txt')
	for l in f1: 
		cls_ = l.split(';')[4].split('\n')[0] 
		if cls_ in classes:
			continue
		else:
			classes[cls_] = 1

	f1.close()
f.close()
print classes.keys(), len(classes.keys())