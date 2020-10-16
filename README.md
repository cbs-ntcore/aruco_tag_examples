A few example scripts for working with
[Aruco tags](https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html) in 
opencv, python and (sometimes) on a raspberry pi

To generate an image containing tags run generate_tags.py

To detect tags run find_tags.py. This script takes many command line arguments
(run "python3 find_tags.py -h" to see the available options). Some examples include:

```bash

# load example.jpg and detect tags after setting the detector parameter
# minMarkerPerimeterRate to 0.01 and show the image with the detected tags
# downsampled to show only every 4th pixel
python3 find_tags.py example.jpg -d 4 -p minMarkerPerimeterRate=0.01

# take the live video feed from the first available opencv video
# source (camera) on your computer and run live tag detection
python3 find_tags.py camera

# take the live video feed from an attached picamera (only works on a
# Raspberry Pi) after setting the resolution to 4032x3040 pixels
# (this resolution will only work for the High Quality camera)
python3 find_tags.py picamera -s resolution=4032x3040
```
