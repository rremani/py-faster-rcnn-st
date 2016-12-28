#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

# Classes for the required problem
CLASSES = ('__background__', 'Auto Parking', 'Bump Ahead', 'Bus Stop Ahead', 'Car And Taxi Parking', 'Compulsory Keep Left', 'Cross Road',
                         'Cycle Crossing', 'Divide Two Side', 'Eating Place Ahead', 'Gap In Median', 'Go Slow_red', 'Guarded Railway Crossing',
                         'Handicapped Parking', 'Horn Prohibited', 'Hospital Ahead', 'Left Hand Curve', 'No Parking_blue', 'No Parking_rb', 'No Parking_red',
                         'No Stopping And No Parking', 'Parking Ahead', 'Parking Ahead_even', 'Parking Ahead_odd', 'Parking Both Sides', 'Parking On This Side',
                         'Pedestrian Crossing', 'Petrol Pump', 'Petrol Pump Ahead', 'Public Telephone Ahead', 'Right Hand Curve', 'School Ahead',
                         'School Ahead_blue', 'Scooter And Motor Cycle Parking', 'Side Road Left', 'Speed Breaker', 'Stop', 'T Junction', 'U Turn Allow_blue',
                         'Speed Limit 30', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 100', 'Speed Limit 120')

# Models from which classes are going to be predicted
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
		'inria': ('INRIA_Person',
				'vgg_cnn_m_1024_rpn_stage1_iter_20000.caffemodel'),
		'gtsdb': ('gtsdb',
				'3gtsdb_iter_40000.caffemodel'),
		# 'mmi': ('mmi',
		# 		'mmi_26_10_iter_60000.caffemodel')}
        'mmi': ('mmi',
              'mmi_15_12_iter_2000.caffemodel'),
        'transfer': ('mmi',
              'mmi_16_12_transfer_6_5000_iter_5000.caffemodel')}


def vis_detections(im, class_name, dets, image_name,out_file,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        # print "no detections  of this class found!:("
        return np.array([])
    f=open(out_file,'a')
    im = im[:, :, (2, 1, 0)]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    output = np.empty((0,5))
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=1.5))
        ax.text(bbox[0], bbox[1] - 2,'{:s} {:.3f}'.format(class_name, score),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white')
        row = np.array(np.append(bbox,class_name),ndmin = 2)
        output = np.append(output,row, axis = 0)
        f.write(str(image_name)+';'+str(bbox)+';'+str(class_name)+';'+str(score)+'\n')
    ax.set_title(('{} detections with ''p({} | box) >= {:.1f}').format(class_name, class_name,thresh),fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # name=image_name.split('.')[0][len[name]-6:]
    name=image_name
    try:
        os.mkdir('output_detections_transfer_retrain')
    except:
        pass
    plt.savefig('output_detections_transfer_retrain/'+name+'_'+class_name+".png")
    plt.close()
    # plt.draw()
    f.close()
    # print "uoptu: ", output
    return output

#def demo(net, image_name,path,out_file):
def demo(net, image_name,out_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(path, image_name)
    # im_file = image_name[:-4] #*
    #im_file

    if os.path.isfile(im_file+'.png'):
        im = cv2.imread(im_file+'.png')
    elif os.path.isfile(im_file+'.jpg'):
        im = cv2.imread(im_file+'.jpg')
    else:
        return 0

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    # print "scores shape:",scores.shape
    # print "boxes shape:",boxes.shape
    # print "scores:",scores
    timer.toc()
    #print np.amax(scores, axis=1)
    #print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    output = np.empty((0,5))
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        temp = vis_detections(im, cls, dets, image_name,out_file,thresh=CONF_THRESH)
        if(temp.size != 0):
        	output = np.append(output, temp,axis=0)
    
    return output
    


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--path',dest='path',help='Path the test images folder',default='/home/ce/Documents/py-faster-rcnn/data/MMI/data/Images2/')
    parser.add_argument('--file',dest='test_file',help='Path the test images txt file')
    parser.add_argument('--out_file',dest='out_file',help='Path the results')

    args = parser.parse_args()

    return args

INCORRECT = -1
def sortInternalClasses(classes):
	newClasses = np.array([])
	singleClass = []
	# assuming each row in classes is of the shape : "w,x,y,z,class_label"
	for cl in np.unique(classes[:,4]):
		index = np.where(classes[:,4] == cl)[0]
		singleClass = classes[[x for x in index],:]
		singleClass = singleClass[singleClass[:,0].argsort()]
		classes = np.delete(classes, ([x for x in index]), axis=0)
		if(newClasses.size == 0):
			newClasses = singleClass
		else :
			newClasses = np.concatenate((newClasses,singleClass),axis= 0)
		singleClass = []
	return newClasses

def checkAccuracy(annotations, image_detections):
    annotations = sortInternalClasses(np.array(annotations))
    image_detections = sortInternalClasses(image_detections)
    annotations[:,0:4].astype(float)
    image_detections[:,0:4].astype(float) 
    print " "
    print " "
    print 'annotations: ', annotations
    print 'image_detections: ',image_detections 
    if(len(image_detections) > len(annotations)):
    	return INCORRECT
    for cl in np.unique(annotations[:,4]):
    	pred = image_detections[np.where(image_detections[:,4] == cl)[0],:]
    	true = annotations[np.where(annotations[:,4] == cl)[0],:]
    	print "pred: ",pred
    	print "true: ",true
    	if(len(pred) != len(true)):
    		return INCORRECT
    	error = np.mean( true != pred )
    	print 'error: ',error
    	if (error > 10): 
    		print "high inaccuracy"
    		return INCORRECT
    return 0


# def softmax_loss(x, y):
#   """
#   Computes the loss and gradient for softmax classification.
#   Inputs:
#   - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
#     for the ith input.
#   - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
#     0 <= y[i] < C
#   Returns a tuple of:
#   - loss: Scalar giving the loss
#   - dx: Gradient of the loss with respect to x
#   """
#   probs = np.exp(x - np.max(x, axis=1, keepdims=True))
#   probs /= np.sum(probs, axis=1, keepdims=True)
#   N = x.shape[0]
#   loss = -np.sum(np.log(probs[np.arange(N), y])) / N
#   dx = probs.copy()
#   dx[np.arange(N), y] -= 1
#   dx /= N
# return loss, dx

def get_annotations(image_name):
    testAnnotationsPath = os.path.join(cfg.DATA_DIR, 'MMI','data','test_Annotations')
    f = open(testAnnotationsPath+"/"+image_name+".txt")
    annotations =  []
    for line in f:
        row = line.split(';')
        row[4] = row[4].split('\n')[0]
        annotations.append(row)
    return annotations

from PIL import Image
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR,'..', NETS[args.demo_net][0],'faster_rcnn_transfer_fcn', 'test.prototxt')
    # prototxt = os.path.join(cfg.MODELS_DIR,'..', NETS[args.demo_net][0],'faster_rcnn_end2end', 'test.prototxt')
	#    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')

    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models','transferLearningTESTING',
                              NETS[args.demo_net][1])
    # caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                           NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    #path='/home/rishabh/py-faster-rcnn/data/demo'
    if args.test_file:
        f1=open(args.test_file,'r')
        data=f1.read()
        im_names=np.unique(data.split('\n'))
        #print im_names
        path = os.getcwd()+'/data/MMI/data/Images/'
    else:
        im_names=sorted(os.listdir(args.path))
        path=args.path

    total=len(im_names)
    a=0
    #demo(net, '/home/ce/Documents/py-faster-rcnn/data/MMI/data/Images2/00000.png',args.out_file)
    for layer in net.params:
        print layer
        print 'weights: ',net.params[layer][0].data
        print 'bais: ', net.params[layer][1].data

    for im_name in im_names:
        print im_name
        #demo(net, '/home/ce/Documents/py-faster-rcnn/data/MMI/data/Images2/00000.png',args.out_file)
        output_detections = np.array((0,5))
        try:
            output_detections= demo(net,im_name,args.out_file)
        except:
            continue
        annotations = get_annotations(im_name)
        print annotations
        if(checkAccuracy(annotations, output_detections) < 0):
            print "modifying parameters on image : ", im_name
            # net.blobs['cls_score'].diff[0]['target_class']=1
            i = 0
            for row in annotations:
                print row[-1]
                cls_ind =  CLASSES.index(row[-1])
                print cls_ind
                net.blobs['cls_prob'].diff[i][cls_ind] = 1
                net.blobs['bbox_pred'].diff[i][4*cls_ind:4*(cls_ind+1)] = row[0:4]
                i+=1
            # print net.blobs['cls_prob'].diff[0][39]
            # print net.blobs['bbox_pred'].diff[0][4*39:4*40]
            net.backward(**{net.outputs[1]: net.blobs['cls_prob'].diff, net.outputs[0]:net.blobs['bbox_pred'].diff})
            # print 'cls-prob: diff:',net.blobs['cls_prob'].diff
            # print 'bbox_pred-new: diff:',net.blobs['bbox_pred'].diff
            # print 'fc6 diff', net.blobs['fc6'].diff
            learning_rate = 0.01
            for layer in net.params:
                print layer
                net.params[layer][0].data[...] -= learning_rate*net.params[layer][0].diff
                net.params[layer][1].data[...] -= learning_rate*net.params[layer][1].diff
            print "need to backward propagate on this image"
            
        print a,'/',total,'Demo for data/demo/{}'.format(im_name)
        a+=1
        os.getcwd()
    # x = raw_input("")    
    for layer in net.params:
        print layer
        print 'weights: ',net.params[layer][0].data
        print 'bais: ', net.params[layer][1].data
    print os.getcwd()
    net.save("./retrained_Models/mmi_retrained_23_12.caffemodel")
       #plt.savefig('output/'+ im_name)
       # plt.show()
       # continue

		# #TBD



#     os.system()
'''
for layer in net.layers:
    for blob in layer.blobs:
        blob.data[...] -= blob.diff
print net._layer_names[17],net._layer_names[8]

print "some layer:",net.layers[45].blobs[0].diff

# prob_diff = rand(net.blobs('cls-prob').shape)
'''
# net.backward(diff=net.blobs['cls_prob'].data)
# net.backward()

#./tools/train_net.py --gpu 0 --solver models/mmi/faster_rcnn_transfer_fcn/solver.prototxt --imdb mmiTransfer_train --cfg experiments/cfgs/faster_rcnn_end2end.yml --weights data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel --iters 2000 >> ./transferTesting/out_mmi_transfer_16_12_1
            # solverProto = os.path.join(cfg.MODELS_DIR,'..', NETS[args.demo_net][0],'faster_rcnn_transfer_fcn', 'solver.prototxt')
            # solver = caffe.get_solver(solverProto)
            # solver.step(1)