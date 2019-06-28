# image_feature_finder
opencv python w basler camera ellipse like objects finding helper tools
  - can identify ellipse like objects, for example: beans

required:
1. pyplon camera
2. python 3 w opencv2
3. ubuntu os

startup:
1. clone script to working_dir/
2. add new dir images to working dir as: working_dir/images/
3. run python working_dir/feature_finder.py h


log:
1. get image edges
2. fill within the edges and some other util
3. identify contours based on hsv mask
4. multiple hsv color ranges to indetify beans
5. view mode only now possible
6. make snapshots in view mode
7. tested with camera
8. run with images out of a directory
9. pick a color from image at runtime and display corresponding mask
10. save the new feature in a json file
11. load features from json fill  
12. can change image dimensions runtime
13. track difference between images and prevent unneeded processing
14. hsv value  slider with arrow keys
15. multithreading mode (nodisplay, seperate lists using locks)
16. added custom image buffer
17. 

Todo:
 
1. add watershed to separate colling beans
2. method to add new solutions
3. update base images routine
4. faster image sharing using semafores
