# To use these tools

1. First convert your output into the same form as sample results

2. To Find all signs of a certain class and split them into training and test set
   python filterAnnotationFile.py -f stop -p stop 80 allTrainingAnnotations.csv

3. To extract the training images
   python extractAnnotations.py crop stop-split1.csv 


4. To evaluate detections we use the below command with the output result

   python evaluateDetections.py results.csv stop-split2.csv 
------
Number of annotations:	236
------
Testing with a Pascal overlap measure of: 0.50
True positives:		212
False positives:	512
False negatives (miss):	24
------
Precision:		0.2928
Recall:			0.8983


5. Finally we generate a PR curve from the results and ground truth annotations

   python generatePRC.py -d results1.csv results2.csv results3.csv -gt stop-split2.csv -o
  
   for multiple plotting:
   python ~/phd/code/annotationTools/generatePRC.py -d results1.csv results2.csv results3.csv -d otherResults1.csv otherResults2.csv otherResults3.csv otherResults4.csv -gt stop-split2.csv -t "PR curve for stop sign detection" -l "My new great detector" "Another detector" -o

6. Misc commands:

$ python generatePRC.py -h
usage: generatePRC.py [-h] [-gt annotations.csv]
                      [-d detections.csv [detections.csv ...]] [-t "PRC plot"]
                      [-l ["Team, algorithm" ["Team, algorithm" ...]]]
                      [-p 0.5] [-o] [-s prcPlot.png] [--noInterpolation]

Generate a precision-recall curve and compute the area under the curve (AUC)
from multiple detection results and an annotation file.

optional arguments:
  -h, --help            show this help message and exit
  -gt annotations.csv, --groundTruth annotations.csv
                        The path to the csv-file containing ground truth
                        annotations.
  -d detections.csv [detections.csv ...], --detectionPaths detections.csv [detections.csv ...]
                        Paths to multiple the csv-files containing detections.
                        Each line formatted as filenameNoPath;upperLeftX;upper
                        LeftY;lowerRightX;lowerRightY. No header line. The
                        files should be produced with different parameters in
                        order to create multiple precision/recall data points.
                        This flag can be given several times to plot multiple
                        detector agains the ground truth.
  -t "PRC plot", --title "PRC plot"
                        Title put on the plot.
  -l ["Team, algorithm" ["Team, algorithm" ...]], --legend ["Team, algorithm" ["Team, algorithm" ...]]
                        Legend for each curve in the plot. Must have the same
                        number of entries as there are curves if given. If not
                        given, generic titles are used.
  -p 0.5, --pascal 0.5  Define Pascal overlap fraction.
  -o, --plot            Show plot of the computed PR curve.
  -s prcPlot.png, --savePlot prcPlot.png
                        Save the computed PR curve to the file prcPlot.png.
  --noInterpolation     By default the PR curves are interpolated according to
                        Davis & Goadrich "The Relationship Between Precision-
                        Recall and ROC Curves. If this flag is given,
                        interpolation is disabled.
