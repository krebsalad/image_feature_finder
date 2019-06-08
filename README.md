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
1. get image edges
2. fill within the edges
3. identify contours based on hsv mask
4. multiple hsv color ranges to indetify beans
5. view mode only now possible
6. make snapshots in view mode
7. run with images out of a directory
8. pick a color from image at runtime and display corresponding mask
9. save the new feature in a json file
10. load features from json file
11. only requires image queue
12. can change image dimensions runtime
13. track difference between images and prevent unneeded processing
14. hsv color slider with arrow keys


Todo:
 
1. fix threading (output queues not working in no display mode])
2. add watershed to separate colling beans
3. test with actual camera
4. save output to file
