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

'''CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')'''

CLASSES = ('__background__', 'Auto Parking', 'Bump Ahead', 'Bus Stop Ahead', 'Car And Taxi Parking', 'Compulsory Keep Left', 'Cross Road',
                         'Cycle Crossing', 'Divide Two Side', 'Eating Place Ahead', 'Gap In Median', 'Go Slow_red', 'Guarded Railway Crossing',
                         'Handicapped Parking', 'Horn Prohibited', 'Hospital Ahead', 'Left Hand Curve', 'No Parking_blue', 'No Parking_rb', 'No Parking_red',
                         'No Stopping And No Parking', 'Parking Ahead', 'Parking Ahead_even', 'Parking Ahead_odd', 'Parking Both Sides', 'Parking On This Side',
                         'Pedestrian Crossing', 'Petrol Pump', 'Petrol Pump Ahead', 'Public Telephone Ahead', 'Right Hand Curve', 'School Ahead',
                         'School Ahead_blue', 'Scooter And Motor Cycle Parking', 'Side Road Left', 'Speed Breaker', 'Stop', 'T Junction', 'U Turn Allow_blue',
                         'Speed Limit 30', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 100', 'Speed Limit 120')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
		'inria': ('INRIA_Person',
				'vgg_cnn_m_1024_rpn_stage1_iter_20000.caffemodel'),
		'gtsdb': ('gtsdb',
				'3gtsdb_iter_40000.caffemodel'),
		'mmi': ('mmi',
				'mmi_26_10_iter_60000.caffemodel')}


def save_blobs(bbox,img,name,class_name): #here labels should be a list of label
     left=int(bbox[0])
     top=int(bbox[1])
     right=int(bbox[2])
     bottom=int(bbox[3])
     crop_img=img[top:bottom, left:right]
     crop_img=cv2.resize(crop_img, (100, 100))
     cv2.imwrite('/home/ce/Documents/py-faster-rcnn/blobs/'+name+'_'+class_name+".jpg",crop_img)

def get_overlap(l1,l2):
    l1,l2=l1.tolist(),l2.tolist()
    xa1,ya1,xa2,ya2=l1[0],l1[1],l1[2],l1[3]
    xb1,yb1,xb2,yb2=l2[0],l2[1],l2[2],l2[3]
    dx =min(xa2, xb2) -max(xa1, xb1)
    #print xa2, xb2,xa1, xb1
    dy = min(ya2, yb2) - max(ya1, yb1)
    #print dx,dy
    area_a=(xa2-xa1)*(ya2-ya1)
    area_b=(xb2-xb1)*(yb2-yb1)
    if (dx>=0) and (dy>=0):
        area_i= dx*dy
        area=area_a+area_b-area_i
        return float(area_i)/area
    else:
        return 0
def vis_detections(im, class_name, dets, image_name,out_file,thresh=0.5):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]

    if len(inds) == 0:
        return
    out_file=os.path.join('/home/ce/Documents/py-faster-rcnn/',out_file)
    f=open(out_file,'a')



    im1=im
    im = im[:, :, (2, 1, 0)]

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(im, aspect='equal')
    print inds
    s=[]
    for j in inds:
    	score1=dets[j,-1]
    	s.append(score1)

    h=np.zeros((len(inds),len(inds)))
    for k in inds:
        for l in inds:
            bbox1=dets[k, :4]
            bbox2=dets[l, :4]
            area=get_overlap(bbox1,bbox2)
            h[k,l]=area

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        h_area=h[i]

        over_lapped=np.where(h_area>0.2)[0]
        over_lapped = over_lapped.tolist()

        print over_lapped


        s_new = np.array(s)[over_lapped]
        t=np.max(s_new)
        z=np.argmax(s_new)
        max_lapped = over_lapped[z] #index of the max score under lapped regions
        print max_lapped

        if len(over_lapped)==1:

            name=image_name.split('.')[0]

            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='green', linewidth=1.5))
            ax.text(bbox[0], bbox[1] - 2,'{:s} {:.3f}'.format(class_name, score),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white')
            f.write(str(image_name)+';'+str(bbox)+';'+str(class_name)+';'+str(score)+'\n')
            save_blobs(bbox,im1,name,class_name)
        else:

            if score >=t and max_lapped==i:

                name=image_name.split('.')[0]
                save_blobs(bbox,im1,name,class_name)
                ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='green', linewidth=1.5))
                ax.text(bbox[0], bbox[1] - 2,'{:s} {:.3f}'.format(class_name, score),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white')
                f.write(str(image_name)+';'+str(bbox)+';'+str(class_name)+';'+str(score)+'\n')

    ax.set_title(('{} detections with ''p({} | box) >= {:.1f}').format(class_name, class_name,thresh),fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    name=image_name.split('.')[0]
    plt.savefig('/home/ce/Documents/py-faster-rcnn/'+'output_detections/'+name+'_'+class_name+'.jpg')

    #save_blobs(bbox,im1,name,class_name)
    plt.close()
    f.close()

def demo(net, image_name,path,out_file):
#def demo(net, image_name,out_file):

    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(path, image_name[:-4])
    #print im_file
    #im_file = image_name

    if os.path.isfile(im_file+'.png'):
        #im_file = im_file[:-4]
        #print im_file
        im = cv2.imread(im_file+'.png')

    elif os.path.isfile(im_file+'.jpg'):
        im = cv2.imread(im_file+'.jpg')

    else:
        return 0


    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    #print np.amax(scores, axis=1)
    #print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])



    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, image_name,out_file,thresh=CONF_THRESH)


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
    parser.add_argument('--path',dest='path',help='Path the test images folder',default='/home/ce/Documents/py-faster-rcnn/data/MMI/data/test-images/')
    parser.add_argument('--file',dest='test_file',help='Path the test images txt file')
    parser.add_argument('--out_file',dest='out_file',help='Path the results')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    try:
        os.mkdir('output_detections')
    except:
        pass
    #print name
    prototxt = os.path.join(cfg.MODELS_DIR,'..', NETS[args.demo_net][0],'faster_rcnn_end2end', 'test.prototxt')
#    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')

    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

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
        _, _= im_detect(net, im)

    #path='/home/rishabh/py-faster-rcnn/data/demo'

    if args.test_file:
        f1=open(args.test_file,'r')
        data=f1.read()
        im_names=np.unique(data.split('\n'))
        #path = os.getcwd()+'/data/MMI/data/Images/'
        #print path
    else:
        im_names=sorted(os.listdir(args.path))
       # print im_names
        path=args.path

    total=len(im_names)
    a=0
    #print im_names
    for im_name in im_names:
        #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #print im_name
        try:
           # print im_name
            #im_name = im_name[:-4]
            demo(net,im_name,path,args.out_file)

        except:
            continue
        print a,'/',total,'Demo for data/demo/{}'.format(im_name)
        a+=1
       # plt.savefig('output/'+im_name)
       # plt.show()
       # continue
