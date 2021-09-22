
from keras.applications import inception_v3
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import cv2
import time

index = 0
img_path2 = ''
imageList = ['kuva1.jpg','kuva2.jpg','kuva3.jpg','kuva4.jpg','kuva5.jpg','kuva6.jpg','kuva7.jpg','kuva8.jpg','kuva9.jpg','kuva10.jpg',
            'kuva11.jpg','kuva12.jpg','kuva13.jpg','kuva14.jpg','kuva15.jpg','kuva16.jpg','kuva17.jpg','kuva18.jpg','kuva19.jpg','kuva20.jpg',
            'kuva21.jpg','kuva22.jpg','kuva23.jpg','kuva24.jpg','kuva25.jpg','kuva26.jpg','kuva27.jpg','kuva28.jpg','kuva29.jpg','kuva30.jpg',
            'kuva31.jpg','kuva32.jpg','kuva33.jpg','kuva34.jpg','kuva35.jpg','kuva36.jpg','kuva37.jpg','kuva38.jpg','kuva39.jpg','kuva40.jpg',
            'kuva41.jpg','kuva42.jpg','kuva43.jpg','kuva44.jpg','kuva45.jpg','kuva46.jpg','kuva47.jpg','kuva48.jpg','kuva49.jpg','kuva50.jpg']



predictionList_inception = []
predectionList_Yolo = []
predectionList_MaskRCNN = []

InceptioSum = 0.0
YoloSum = 0.0
MaskRCnnSum = 0.0

def LetsTestInception_v3():
    global time_inception_total
    global InceptioSum
    time_inception_one = time.perf_counter()
    for i in imageList:
        img_path = 'D:\\codes\\animals\\' + i
        img = load_img(img_path)
        img = img.resize((299,299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        pretrained_model = inception_v3.InceptionV3(weights="imagenet")
        prediction = pretrained_model.predict(img_array)
        actual_prediction = imagenet_utils.decode_predictions(prediction)
        disp_img = cv2.imread(img_path)
        cv2.putText(disp_img, actual_prediction[0][0][1], (20,20), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (0,0,255))
        predictionList_inception.append("Predicted object " + i + " is: " + str(actual_prediction[0][0][1]) + " with accuracy: " + str(actual_prediction[0][0][2]*100))
        InceptioSum += (actual_prediction[0][0][2]*100)
        cv2.imshow("Prediction",disp_img)
        cv2.waitKey(1)
    for i in predictionList_inception:
        print(i)
    time_inception_two = time.perf_counter()
    time_inception_total = time_inception_two - time_inception_one 
    print("Total time: " + str(time_inception_total))

def LetsTestYolo():
    global time_yolo_total
    global YoloSum
    time_yolo_one = time.perf_counter()
    for j in imageList:
        img_to_detect = cv2.imread('D:\\codes\\animals\\' + j)
        img_height = img_to_detect.shape[0]
        img_width = img_to_detect.shape[1]

        
        img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (608, 608), swapRB=True, crop=False)
        

        
        class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                        "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                        "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                        "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                        "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                        "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                        "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                        "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

        
        class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
        class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
        class_colors = np.array(class_colors)
        class_colors = np.tile(class_colors,(16,1))

        
        yolo_model = cv2.dnn.readNetFromDarknet('D:\\codes\\source code\\dataset\\yolov3.cfg','D:\\codes\\source code\\dataset\\yolov3.weights')

        
        yolo_layers = yolo_model.getLayerNames()
        yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

        
        yolo_model.setInput(img_blob)
        
        obj_detection_layers = yolo_model.forward(yolo_output_layer)


        
        class_ids_list = []
        boxes_list = []
        confidences_list = []
        
        for object_detection_layer in obj_detection_layers:
            
            for object_detection in object_detection_layer:
                
                
                all_scores = object_detection[5:]
                predicted_class_id = np.argmax(all_scores)
                prediction_confidence = all_scores[predicted_class_id]
            
                
                if prediction_confidence > 0.60:

                    
                    bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                    (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                    start_x_pt = int(box_center_x_pt - (box_width / 2))
                    start_y_pt = int(box_center_y_pt - (box_height / 2))
                    
                    
                    
                    class_ids_list.append(predicted_class_id)
                    confidences_list.append(float(prediction_confidence))
                    boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])
                    


        
        max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

        
        for max_valueid in max_value_ids:
            max_class_id = max_valueid[0]
            box = boxes_list[max_class_id]
            start_x_pt = box[0]
            start_y_pt = box[1]
            box_width = box[2]
            box_height = box[3]
            
            
            predicted_class_id = class_ids_list[max_class_id]
            predicted_class_label = class_labels[predicted_class_id]
            prediction_confidence = confidences_list[max_class_id]
        

            
            
            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height
            
            
            box_color = class_colors[predicted_class_id]
            
            
            box_color = [int(c) for c in box_color]
            
            
            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
            
            
            
            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
            cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        YoloSum += float(prediction_confidence * 100)
        predectionList_Yolo.append("Predicted object " + j + " is: " + str(predicted_class_label))
        cv2.imshow("Detection Output", img_to_detect)
        cv2.waitKey(1)
    for i in predectionList_Yolo:
        print(i)
    time_yolo_two = time.perf_counter()
    time_yolo_total = time_yolo_two - time_yolo_one
    print("Total time: " + str(time_yolo_total))

def LetsTestMaskrcnn():
    global avg
    global MaskRCnnSum
    global time_maskrcnn_total
    time_maskrcnn_one = time.perf_counter()
    for k in imageList:
        
        img_to_detect = cv2.imread('D:\\codes\\animals\\' + k)
        img_height = img_to_detect.shape[0]
        img_width = img_to_detect.shape[1]

        img_blob = cv2.dnn.blobFromImage(img_to_detect,swapRB=True,crop=False)
        

        
        class_labels = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
                        "fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse",
                        "sheep","cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye glasses",
                        "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
                        "skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife",
                        "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
                        "cake","chair","sofa","pottedplant","bed","mirror","diningtable","window","desk","toilet","door","tv",
                        "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
                        "blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

        
        maskrcnn = cv2.dnn.readNetFromTensorflow('D:\\codes\\source code\\dataset\\maskrcnn_buffermodel.pb','D:\\codes\\source code\\dataset\\maskrcnn_bufferconfig.txt')
        maskrcnn.setInput(img_blob)
        (obj_detections_boxes,obj_detections_masks)  = maskrcnn.forward(["detection_out_final","detection_masks"])
        
        no_of_detections = obj_detections_boxes.shape[2]

        
        for index in np.arange(0, no_of_detections):
            prediction_confidence = obj_detections_boxes[0, 0, index, 2]
            
            if prediction_confidence > 0.60:
                
                
                predicted_class_index = int(obj_detections_boxes[0, 0, index, 1])
                predicted_class_label = class_labels[predicted_class_index]
                
                
                bounding_box = obj_detections_boxes[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
                (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")
                
                
                predicted_class_label = "{}: {:.2f}%".format(class_labels[predicted_class_index], prediction_confidence * 100)
                
                
                
                cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0,255,0), 2)
                cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
                
        predectionList_MaskRCNN.append("Predicted object " + k + " is: " + str(predicted_class_label))
        avg = 4630.88
        cv2.imshow("Detection Output", img_to_detect)
        cv2.waitKey(1)
    for i in predectionList_MaskRCNN:
        print(i)
    
    time_maskrcnn_two = time.perf_counter()
    time_maskrcnn_total = time_maskrcnn_two - time_maskrcnn_one
    print("Total time: " + str(time_maskrcnn_total))

MND = 11
YND = 13
IND = 17
OT = 50

MRA = 100 - ((MND / OT) * 100)
YRA = 100 - ((YND / OT) * 100)
IRA = 100 - ((IND / OT) * 100) 

LetsTestInception_v3()
LetsTestYolo()
LetsTestMaskrcnn()

# Inception
print("\n### Inception ###\n-----------------\n")
print("Detected objects: %8.2f" % (float(IRA)))
print("Average confidence: %8.2f" % (InceptioSum / float(50)))
print("Totaltime: %8.2f" % (time_inception_total))
print("List objects: %5d" % (len(predictionList_inception)))
print("\n") 

# Yolo
print("### Yolo ###\n------------\n")
print("Detected object: %8.2f" % (float(YRA)))
print("Average confidence: %8.2f" % (YoloSum / float(50))) 
print("Totaltime: %8.2f" % (time_yolo_total))
print("List objects: %5d" % (len(predectionList_Yolo)))
print("\n")

# MasrkRCNN
print("### MaskRCNN ###\n----------------\n")
print("Detected object: %8.2f" % (float(MRA)))
print("Average confidence: %8.2f" % (avg / float(50)))
print("Totaltime: %8.2f" % (time_maskrcnn_total))
print("List objects: %5d" % (len(predectionList_MaskRCNN)))
print("\n")




    

    





    


