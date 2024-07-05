import cv2, time, os, tensorflow as tf
import numpy as np



from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)#for getting reproducable result for all runs

class Detector:
  def __init__(self):
     pass
  
 ###to read class(coco.names) file ###
  def readClasses(self, classesFilePath):#creating readClasses func with classesFilePath argument
      with open(classesFilePath, 'r') as f:#opening in read mode
          self.classesList = f.read().splitlines()#split all the lines and put text in list

        #unique color for every class label
          self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))#initialize color values
          print(len(self.classesList), len(self.colorList))# to check classes and color list are same
        
  ####to download the model #####
  def downloadModel(self, modelURL):#creating downloadModel func with modelURL argument

       fileName = os.path.basename(modelURL)#.basename method is to extract filename from url
       self.modelName = fileName[:fileName.index('.')]#to extract model name without extensions from file name
 
       print(fileName)#prints the filename
       print(self.modelName)#prints the modelName

       self.cacheDir = "./pretrained_models"#self.cache directory

       os.makedirs(self.cacheDir, exist_ok=True)#create pretrained_models dir,if dir exists model is not downloaded again
                                                          
       get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)#to download OD model

##to load the downloaded model
  def loadModel(self):
       print("Loading Model" + self.modelName)#print the message & concatenate model name
       tf.keras.backend.clear_session()#clear keras session
       self.model=tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))#load model in variable self.model
  
       print("Model " + self.modelName + "loaded successfully...") 

  def createBoundigBox(self, image, threshold = 0.5):
       inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
       inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
       inputTensor = inputTensor[tf.newaxis,...]

       detections = self.model(inputTensor)

       bboxs = detections['detection_boxes'][0].numpy()
       classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
       classScores = detections['detection_scores'][0].numpy()

       imH, imW, imC = image.shape

       bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
       iou_threshold=threshold, score_threshold=threshold)

       print(bboxIdx)

       if len(bboxIdx) != 0:
           for i in bboxIdx:
               bbox = tuple(bboxs[i].tolist())
               classConfidence = round(100*classScores[i])
               classIndex = classIndexes[i]

               classLabelText = self.classessList[classIndex].upper()
               classColor = self.colorList[classIndex]

               displayText = '{}: {}%'.format(classLabelText, classConfidence)

               ymin, xmin, ymax, xmax = bbox
  
               xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
               xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

               cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
               cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

               ###########################################################
               lineWidth = min(int((xmax - xmin) * 0.2)), int((ymax - ymin) * 0.2)

               cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
               cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)

               cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
               cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)
               ###################################################################

               cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
               cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

               cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
               cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)
               

               return image

  def predictImage(self, imagePath, threshold = 0.5):
       image = cv2.imread(imagePath)

       bboxImage = self.createBoundigBox(image, threshold)
       
       cv2.imwrite(self.modelName + ".jpg", bboxImage)
       cv2.inshow("Result", bboxImage)
       cv2.waitKey(0)
       cv2.destroyAllWindows()

  def predictVideo(self, videoPath, threshold = 0.5):
       cap = cv2.VideoCapture(videoPath)

       if (cap.isOpened() == False):
          print("Error opening file...")
          return

       (success, image) = cap.read()
  
       startTime = 0

       while success:
            currentTime = time.time()

            fps = 1/(currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundigBox(image, threshold)

            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
            cv2.imshow("Result", bboxImage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()
       cv2.destroyAllWindows()
