## YOLOv3Model
* YOLO v3 model built from scratch using pytorch
* Download weights for COCO and VOC: [from here](https://pjreddie.com/media/files/yolov3.weights)
* update number of classes for custom object detection

### How to create yolo from scrath by just referring config file
##### Please check yolov3.cfg file for configuration of YOLO V3
##### There are 5 types of layers in YOLO V3
* Convolution

* Shortcut: It's a skip connection like the one used in ResNet, where the
    output of the previous layer and and previous 3rd layer backwards from shortcut layer.
    We concate them and squeeze them through linear activation function.
    
    ```
    #Example:   
      [shortcut]
      from=-3  
      activation=linear  
    ```
    
* Upsample: Upsamle is feature map of previous layer with a stride, which helps 
    increasing depth using bilinear upsampling.
    
    ```
    #Example:
     [upsample]
     stride=2
     ```
     
* Route: It has an attribute layers which can have either one, or two values.

    ```
    #Example:
     [route]
     layers = -4
    ```
    
    When layers attribute has only one value, it outputs the feature maps of the 
    layer indexed by the value. In our example, it is -4, so the layer will output 
    feature map from the 4th layer backwards from the Route layer.
    
    ```
    #Example:
     [route]
     layers = -1, 61
    ```
     
    When layers has two values, it returns the concatenated feature maps of the layers
     indexed by it's values. In our example it is -1, 61, and the layer will output 
     feature maps from the previous layer (-1) and the 61st layer, concatenated along 
     the depth dimension.      

* YOLO Detection Layer

    ```
    #Example:
     [yolo]
     mask = 0,1,2
     anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
     classes=80  
     num=9
     jitter=.3
     ignore_thresh = .5
     truth_thresh = 1
     random      
    ```
    
   YOLO layer corresponds to the Detection layer.
    The anchors describes 9 'anchors', but only the anchors which are indexed by 
    'attributes' of the 'mask' tag are used. Here, the value of 'mask' is 0,1,2, which means 
    the first, second and third anchors are used as each cell of the detection layer 
    predicts 3 boxes. In total, we have detection layers at 3 scales, 
    making up for a total of 9 anchors.
