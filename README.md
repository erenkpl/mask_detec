# maskFace


![maske tespit uygulaması](https://user-images.githubusercontent.com/102223932/182337894-8c94e721-615b-42f3-b4fe-ab40bd959e55.png)


Thanks to image processing and object tracking, maskFace recognize faces with mask or without mask. 

## Description

Project come with a new window that we see all the things about mask recognition. In that window, the real time video capture Label came in the middle, and the two side of this label, comes three label right and three label left which has photos that faces with mask or without mask. In right side, added last 3 photos that faces without mask in frequently, also in left side added last 3 photos that faces with mask. In middle label, the live video recording starts as soon as the program starts. And then, program select the face from that frame, and recognize it for with mask or without mask. If the face with mask, than the program add the photo to left side of the window and also save it into the “Face Detection/Masked/” directory, if the face without mask, than the program add the photo to left side of the window and also save it into the “Face Detection/Unmasked/”. But if the face was not recognize, the two side of photo labels stays same and no photos saved in directory. To avoid storage problems, I put a limit to saving system. That limit was allows up to 100 photos for each masked or unmasked directory. So, the project saves at most 200 photos. 

![maskeli_maskesiz](https://user-images.githubusercontent.com/102223932/182338362-069263ef-e581-4db6-872a-cca4f20327c5.png)

![sonuç_maskesiz](https://user-images.githubusercontent.com/102223932/182338436-dd9140ec-b087-41e2-9b02-f50855010b1d.png)


## Getting Started

### Dependencies

* This program needs tensorflow, keras, cuda and opencv. 
* Also needed libraries are; numpy, os, glob, tkinter, PIL (Image, ImageTk), sklearn.metrics(accuracy_score), sklearn.model_selection(train_test_split), sklearn.tree (DecisionTreeRegressor). 
* Windows

### Installing

* You can download from github as .zip.
* for opencv “pip install opencv-python” 
* for tensorflow “pip install tensorflow”
* for keras “pip install keras”

### Executing program

* The only thing to do, run the mask_detection.py from any python IDE. 

## Help

For the best recognition, please use under the enough light. 

## Authors

[@erenkpl] (https://github.com/erenkpl)

## Version History

* 0.2
    * Face Detection folder changed.
    * Save issues solved.
* 0.1
    * Initial Release
