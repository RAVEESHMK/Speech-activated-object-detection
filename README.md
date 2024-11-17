Team Members:
1.Raveesh Mallenhalli Kalleshappa(U13791113)
2.Dharmesh Rahul Kodas Mahadev(U36725202)
3.Nandini Chekuri(U26955877)
# Speech-activated-object-detection
Problem Statement
Computer Vision is a promising to use state-of-art techniques to help people with vision loss. Through this project, we want to explore the possibility of using the hearing sense to understand visual objects. The sense of sight and hearing share a striking similarity: both visual object and audio sound can be spatially localized. It is not often realized by many people that we are capable at identifying the spatial location of a sound source just by hearing it with two ears. In our idea, we build a real-time object detection and position estimation pipeline, with the goal of informing the user about surrounding object and their spatial position using binaural sound.
Methodology
The model uses MobileNet-SSD algorithm for object detection and Google Speech Recognition package (gTTS) for audio processing. Pascal Voc dataset is used for object detection; segmentation and captioning which has 20 different classes making it very versatile for object detection. 

Coco.names conatains the trained dataset which should be kept in the same directory

trust_worthy_ai.py is the code implementation should be ran after installing necessary packages.


To implement in your local machine follow below changes:
1. Line 3:
LABELS = open("D:/raveesh/yolo-master/yolo-coco/coco.names").read().strip().split("\n")
Please change the directory path to where you store coco.names file

2.Line 12:
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")

//redirect to your respective paths for yolov3.weights and yolov3.cfg files

Feel free to contact rm159@usf.edu

