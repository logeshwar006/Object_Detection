from Detector import *


modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"

classFile = "coco.names"#defining the classFile
imagePath = "test/0.jpg"
videoPath = 0 # for webcam
threshold = 0.5

detector = Detector()#initialize detector class and creating object, detector
detector.readClasses(classFile)#calling read classes method,as classFile as parameter
detector.downloadModel(modelURL)#calling downloadModel method, as modelURL as parameter
detector.loadModel()
detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)