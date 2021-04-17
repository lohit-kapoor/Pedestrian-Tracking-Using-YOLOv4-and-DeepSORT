# Pedestrian-Tracking-Using-YOLOv4-and-DeepSORT

To run the code on Google Colab, follow instructions as below:

1. Clone repository for deepsort with yolov4:
'''
!git clone https://github.com/lohit-kapoor/Pedestrian-Tracking-Using-YOLOv4-and-DeepSORT.git
'''
    
2. Step into the Pedestrian-Tracking-Using-YOLOv4-and-DeepSORT folder:
    %cd Pedestrian-Tracking-Using-YOLOv4-and-DeepSORT

3. Install Dependencies:
    !pip install -r requirements-gpu.txt

4. Get YOLOv4 Pre-trained Weights:
    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data/
    
5. Convert YOLOv4 Darknet Weights to TensorFlow model:
    !python save_model.py --model yolov4
    
6. Running DeepSort with YOLOv4:
    !python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/outputs.avi --model yolov4 --dont_show --info
    
