# bean_finder
opencv python w basler camera bean finding helper tools

features:
get image edges
fill within the edges
identify contours based on hsv mask
multiple hsv color ranges to indetify beans
view mode only
make snapshots
run with images
pick a color from image at runtime and display corresponding mask
save the new feature in a json file
load features from json file
only requires image queue
can change image dimensions runtime
track difference between images and prevent unneeded processing


Todo:
 
1. fix threading
2. add watershed to separate colling beans
