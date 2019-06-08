# image_feature_finder
opencv python w basler camera bean finding helper tools

required:
1. pyplon camera
2. python 3 w opencv2
3. ubuntu os

startup:
1. clone script to working_dir/
2. add new dir images to working dir as: working_dir/images/
3. run python working_dir/feature_finder.py h


features:
get image edges
fill within the edges
identify contours based on hsv mask
multiple hsv color ranges to indetify beans
view mode only now possible
make snapshots in view mode
run with images out of a directory
pick a color from image at runtime and display corresponding mask
save the new feature in a json file
load features from json file
only requires image queue
can change image dimensions runtime
track difference between images and prevent unneeded processing


Todo:
 
1. fix threading
2. add watershed to separate colling beans
