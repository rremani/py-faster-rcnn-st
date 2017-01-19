# Basic Commands

* Generate mAP ( Mean Average Precision: Area under Precision Recall curve)

* Check if ImageSets folder contains test.txt with image names and Anotations folder contains the respective ground truth data.

* The following command is then run to produce mAP: ./tools/test_net.py --gpu 0 --def /home/ce/Documents/py-faster-rcnn/models/gtsdb/faster_rcnn_end2end/test.prototxt  --imdb gtsdb_test --cfg experiments/cfgs/faster_rcnn_end2end.yml --net /home/ce/Documents/py-faster-rcnn/data/faster_rcnn_models/gtsdb_v1_iter_60000.caffemodel  --comp --vis
