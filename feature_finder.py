# author:
# Johan Mgina
# Date modified: 8-6-2019
# version: 0.9.7

from pypylon import pylon
import cv2
import time
import sys
import numpy as np
import threading
import queue
import re
import json
import os
import glob

## image sharing ##
image_queue = queue.Queue()
exit_threads = False

# display queue
queue1 = queue.Queue()
queue2 = queue.Queue()
q_size = 3

# acqusition
def initCamera():
    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    #set the dimentions og the image to grab
    camera.Open()
    camera.Width.Value = camera_w  # 0.8% max width of Basler puA2500-14uc camera
    camera.Height.Value =  camera_h# 0.8% max height of Basler puA2500-14uc camera
    camera.OffsetX.Value = 518
    # camera.AcquisitionFrameRate.SetValue(14)

    # set features of camera
    camera.ExposureTime.Value = 110000
    camera.ExposureAuto.SetValue('Off')
    camera.BalanceWhiteAuto.SetValue('Off')
    camera.LightSourcePreset.SetValue('Tungsten2800K')
    camera.GainAuto.SetValue('Off')
    # pylon.FeaturePersistence.Save("test.txt", camera.GetNodeMap())

    print("Using device: ", camera.GetDeviceInfo().GetModelName())
    print("width set: ",camera.Width.Value)
    print("Height set: ",camera.Height.Value)

    # The parameter MaxNumBuffer can be used to control the count of buffers
    # allocated for grabbing. The default value of this parameter is 10.
    camera.MaxNumBuffer = 5

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return camera, converter

def grabImagesThread(camera, converter):
    while camera.IsGrabbing() and not exit_threads:
        # grab image when queue is not full
        if(image_queue.qsize() < q_size):
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img = image.GetArray()
                image_queue.put(img)
            grabResult.Release()

    print("camera stopped!!")

def showScaledImage(window_name, image, down_scale=True):
    rezised_image = image.copy()
    height, width = rezised_image.shape[0:2]
    if(down_scale):
        rezised_image = cv2.resize(rezised_image, ((int)(width/3)+1, (int)(height/3)+1))
    cv2.imshow(window_name, rezised_image)

def displayQueueThread(win_name, queue):
    while(True and not exit_threads):
        if not (queue.empty()):
            display = queue.get(block=True, timeout=None)
            showScaledImage(win_name, display)

        if(cv2.waitKey(10) == 1048603):
            cv2.destroyWindow(win_name)
            break


## find ellipse like objects using hsv colors ##
class FeatureFinder:
    # main routine functions
    def __init__(self, mode, image_queue, display_queue_1=None, display_queue_2=None, height=1554, width=2074, name="feature_finder", verbose=False, max_contours=50):
        # setup
        self.name = name
        self.mode = mode
        self.image_queue = image_queue
        self.display_queue_1 = display_queue_1
        self.display_queue_2 = display_queue_2
        self.verbose = verbose

        #paths
        self.current_dir = os.getcwd()
        self.images_dir = self.current_dir + "/images"
        self.save_file = self.current_dir + "/features.json"

        # default vals
        self.height = height
        self.width = width
        self.features = []
        self.identifiedContours = []
        self.max_contours = max_contours
        self.delay = 0
        self.time_between_images = 0
        self.down_scale_for_display = 3
        self.snapshot_counter = 0
        self.display_height = ((int)(self.height/self.down_scale_for_display)) + 1
        self.display_width = ((int)(self.width/self.down_scale_for_display)) + 1

        # const images
        self.base_image = np.zeros((self.height,self.width,3), np.uint8)
        self.base_image_s = np.zeros((self.height,self.width,1), np.uint8)

        # changing images
        self.current_image = self.base_image.copy()
        self.image_hsv = cv2.cvtColor(self.base_image.copy(),cv2.COLOR_BGR2HSV)

        # processing images
        self.image_result = self.base_image.copy()
        self.image_edges = self.base_image_s.copy()
        self.image_filled = self.base_image_s.copy()

        # kernels
        self.kernel_3x3 = np.array(([-1, -1, -1],[-1, 8, -1],[-1, -1, -1]))

        # print text
        verbose_text = ""
        print_text = ""

        # contious
        if(self.mode[0]):
            self.delay = 1 # 1 ms

        # single
        if not (self.mode[0]):
            self.delay = 0 # wait for input

        # processing setup
        if(self.mode[1]):
            # create new file if not available
            if not (os.path.exists(self.save_file)):
                with open(self.save_file, "w+") as file:
                    file.write("[]")
                    verbose_text += self.name+": created file "+self.save_file + "\n"

            # load data
            with open(self.save_file) as file:
                    self.features = json.load(file)

            # print text
            self.num_of_feature_types = len(self.features)
            print(self.name+": succesfully loaded features")
            verbose_text += self.name+":Features:"+str(self.features) + "\n"

        # display and other features if display allowed
        if not (self.mode[3]):
            # pick a color from a hsv image and create mask
            if(self.mode[2]):
                # create mask
                self.color_picker_image = cv2.resize(self.image_hsv.copy(), (self.display_width, self.display_height))
                self.color_picker_mask = cv2.resize(self.base_image_s.copy(), (self.display_width, self.display_height))

                # color picker vals
                self.clicked_pixel = [0,0,0]
                self.color_picker_range_low = [0,0,0]
                self.color_picker_range_high = [0,0,0]
                self.color_picker_range = [10,10,40]

                # display images
                self.displayImage("Mask Result", self.color_picker_mask, down_scale=False)
                self.displayImage("HSV_color_picker", self.color_picker_image, down_scale=False)

                # callback on click
                cv2.setMouseCallback("HSV_color_picker", self.displayMaskFromChosenPixel)

            # make image when running camera
            if(self.mode[0] and not self.mode[1] and not self.mode[2]):
                self.delay = 1 # 1 ms

            # pick_color mode when running camera
            if(self.mode[0] and self.mode[2]):
                self.delay = 1000 # 1 s

            # print input
            print_text += self.name+" input list:\n"
            if (self.mode[2]):
                print_text += "  : click on a pixel on window 'HSV_color_picker' to create a hsv image mask\n"
                print_text += "  : use arrow keys left and right to change hsv range Value for the mask"
                print_text += "  : use arrow keys up and down to change hsv ranges HUE and Saturation for the mask"
                print_text += "  : press <spacebar> to add ranges feature\n"

            if (self.mode[0] and not self.mode[1] and not self.mode[2]):
                print_text += "  : press <spacebar> to make image\n"

            print_text += "  : press <esc> to exit\n"

            if(self.delay == 0):
                print_text += "  : press any other key to continue to next image\n"
        else:
            if not (display_queue_1):
                print(self.name+" no display queue for resulting images!!")
            if not (display_queue_2):
                print(self.name+" no display queue for original images!!")
            verbose = False

        # print info
        verbose_text += self.name+": set delay to "+ str(self.delay) + "\n"
        if(self.verbose):
            print(verbose_text)


        print(self.name+": succesfully initialised\n" + print_text)

    def run(self):
        # for util
        tick_start = 0
        tick_end = 0

        # process images
        while(True):
            # get and edit image
            if not (self.image_queue.empty()):
                # get image and set data
                self.current_image = self.image_queue.get(block=True, timeout=None)

                # set values if image sizes are differrent
                if(self.current_image.shape[0] != self.height or self.current_image.shape[1] != self.width):
                    self.height, self.width = self.current_image.shape[0:2]
                    self.display_height = ((int)(self.height/self.down_scale_for_display)) + 1
                    self.display_width = ((int)(self.width/self.down_scale_for_display)) + 1
                    self.image_hsv = cv2.cvtColor(self.current_image.copy(),cv2.COLOR_BGR2HSV)
                    self.base_image = np.zeros((self.height,self.width,3), np.uint8)
                    self.base_image_s = np.zeros((self.height,self.width,1), np.uint8)

                # process
                if (self.mode[1]):
                    print(self.name+": processing image...")

                    # timer
                    tick_start = cv2.getTickCount()

                    # process image
                    self.solution1()

                    # print data
                    tick_end = cv2.getTickCount()
                    self.time_between_images = (tick_end - tick_start)/ cv2.getTickFrequency()
                    print(self.name+": took "+str(self.time_between_images)+" to process image")

                # add to display queue
                if (mode[3]):
                    if(self.display_queue_1):
                        self.display_queue_1.put(self.image_result.copy())
                    if(self.display_queue_2):
                        self.display_queue_2.put(self.current_image.copy())

            # display images and handle input
            if (not mode[3]):
                # update mask
                if(self.mode[2]):
                    self.updateColorPickerDisplay()

                if (self.mode[1]):
                    self.displayImage("found_features", self.image_result.copy())

                # always display original image
                self.displayImage("original", self.current_image.copy())

                # handle input / delay / read keys
                if not (self.handleInput()):
                    print(self.name+": exiting..")
                    return

    def updateColorPickerDisplay(self):
        self.image_hsv = cv2.cvtColor(self.current_image.copy(),cv2.COLOR_BGR2HSV)
        self.color_picker_image = cv2.resize(self.image_hsv.copy(), (self.display_width, self.display_height))
        self.color_picker_range_high = [self.clicked_pixel[0] + self.color_picker_range[0], self.clicked_pixel[1] + self.color_picker_range[1], self.clicked_pixel[2] + self.color_picker_range[2]]
        self.color_picker_range_low = [self.clicked_pixel[0] - self.color_picker_range[0], self.clicked_pixel[1] - self.color_picker_range[1], self.clicked_pixel[2] - self.color_picker_range[2]]
        self.color_picker_mask = cv2.inRange(self.color_picker_image.copy(),np.array(self.color_picker_range_low),np.array(self.color_picker_range_high))
        self.displayImage("HSV_color_picker", self.color_picker_image, down_scale=False)
        self.displayImage("Mask Result", self.color_picker_mask, down_scale=False)

    # input
    def handleInput(self):
        key = -1

        # repeat for next keys
        while(key == -1):
            key = -2
            # delay get get input
            key = cv2.waitKeyEx((int)(self.delay))

            # spacebar
            if not (self.mode[3]):
                if (key == 32):

                    # add bean
                    if(self.mode[2]):
                        # get input
                        print(self.name+": Name of to identify feature:")
                        in_str = str(input())

                        # text
                        new_bean = [in_str, self.color_picker_range_low, self.color_picker_range_high]
                        self.features.append(new_bean)

                        # write data
                        text = []
                        text.append("[")
                        for i, bean_type in enumerate(self.features):
                            text.append("\n"+str(bean_type))
                            if(i + 1 != (len(self.features))):
                                text[i + 1] += ","
                        text.append("]")

                        # write file
                        with open(self.save_file, "w") as file:
                            for txt in text:
                                txt = re.sub("'", '"',txt)
                                file.write(txt)

                        print(self.name+": saved features at "+self.save_file)
                        key = -1

                    # make snapshot
                    if(self.mode[0] and not self.mode[1] and not self.mode[2]):
                        self.snapshot_counter +=1
                        pth = self.images_dir+"/image"+ str(self.snapshot_counter) + ".bmp"
                        if(os.path.exists(self.images_dir+"/")):
                            cv2.imwrite(pth,self.current_image)
                            print(self.name+": image made and saved in: " + pth)
                        else:
                            print(self.name+": could not find dir: "+ self.images_dir)

                # arrow key up
                if(key == 65362):
                    if(self.mode[2]):
                        self.color_picker_range[0] += 5
                        self.color_picker_range[1] += 5
                        self.updateColorPickerDisplay()
                        print(self.name+": new hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))

                # arror key down
                if(key == 65364):
                    if(self.mode[2]):
                        self.color_picker_range[0] -= 5
                        self.color_picker_range[1] -= 5
                        self.updateColorPickerDisplay()
                        print(self.name+": hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))

                # arror key right
                if(key == 65363):
                    if(self.mode[2]):
                        self.color_picker_range[2] += 10
                        self.updateColorPickerDisplay()
                        print(self.name+": hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))

                # arror key left
                if(key == 65361):
                    if(self.mode[2]):
                        self.color_picker_range[2] -= 10
                        self.updateColorPickerDisplay()
                        print(self.name+": hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))

            # escape
            if (key == 1048603):
                return False
        return True

    def displayMaskFromChosenPixel(self, event, x, y, f, p):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_pixel = self.color_picker_image[y,x]
            self.updateColorPickerDisplay()
            print(self.name+": hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))

    # solutions
    def solution1(self):
        # enhancement
        img_b_gray = cv2.split(self.current_image.copy())[0]

        # edges
        new_edges_image = self.getImageEdges(img_b_gray, dilation=2, lower=20, upper=50)

        # get difference
        diff_value = -1
        diff_image = self.base_image_s.copy()
        if(self.image_edges.shape[0] == self.height and self.image_edges.shape[1] == self.width):
            diff_image = cv2.bitwise_xor(self.image_edges.copy(), new_edges_image.copy())
            diff_value = len((np.where(diff_image == 255))[0])


        # identify if difference is high
        if(diff_value > 50000 or diff_value == -1 or len(self.identifiedContours) == 0):
            # canny
            self.image_edges = new_edges_image.copy()

            # fill in holes
            self.image_filled = self.fillWithinEdges(self.image_edges.copy())

            # print / display
            if(self.verbose):
                print(self.name +": Difference value = "+str(diff_value))
                self.displayImage("image_gray", img_b_gray.copy())
                self.displayImage("image_edges", self.image_edges.copy())
                self.displayImage("image_filled", self.image_filled.copy())
                self.displayImage("diff_between_consecutive_images", diff_image, wait=True)
                cv2.destroyWindow("image_gray")
                cv2.destroyWindow("image_edges")
                cv2.destroyWindow("image_filled")
                cv2.destroyWindow("diff_between_consecutive_images")

            # mark beans
            self.identifyFilledImageUsingHsvRanges()

        else:
            # print / display
            print(self.name +": skipping image")
            if(self.verbose):
                print(self.name +": Difference value = "+str(diff_value))
                self.displayImage("image_gray", img_b_gray.copy())
                self.displayImage("image_edges", self.image_edges.copy())
                self.displayImage("image_filled", self.image_filled.copy())
                self.displayImage("diff_between_consecutive_images", diff_image, wait=True)
                cv2.destroyWindow("image_gray")
                cv2.destroyWindow("image_edges")
                cv2.destroyWindow("image_filled")
                cv2.destroyWindow("diff_between_consecutive_images")

            # redraw if difference is low
            self.drawIdentifiedContours()

    # feature extraction
    def identifyFilledImageUsingHsvRanges(self):
        # data
        self.identifiedContours.clear()
        self.num_of_feature_types = len(self.features)

        # find contours
        ret, thresh = cv2.threshold(self.image_filled, 100, 200, 3)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if(len(contours) > self.max_contours):
            self.image_result = self.current_image.copy()
            print(self.name +": too many contours "+str(len(contours)) + ", skipping...")
            return

        # are there feature available
        if(self.num_of_feature_types == 0):
            print(self.name +": No features available!!!")
            self.image_result = self.base_image.copy()
            return

        # hsv image
        self.image_hsv = cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGR2HSV)

        # create bean masks
        hsv_image_masks = [None]*self.num_of_feature_types
        for i in range(0, self.num_of_feature_types):
            mask = cv2.inRange(self.image_hsv, np.array(self.features[i][1]), np.array(self.features[i][2]))
            hsv_image_masks[i] = mask

        # identify contours
        self.image_result = self.current_image.copy()
        counter = 0
        print(self.name +":\n")
        for cnt in contours:
            # to print
            print_text = ""
            verbose_text = ""

            # get usefull data
            M = cv2.moments(cnt)
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])

            # filter contours
            if((int)(len(cnt)) < 30):
                continue

            if(centroid_x > self.width and centroid_y > self.height):
                continue

            # redraw contour in a threshold image
            countour_area_image = self.getContourAreaImage(cnt)
            countour_area_image = cv2.bitwise_not(countour_area_image)

            # get num of pixels for each mask
            num_of_pixels = [None]*self.num_of_feature_types
            res_images = [None]*self.num_of_feature_types
            verbose_text += "  num of pixels on ranges:\n"
            for i in range(0, self.num_of_feature_types):
                num_of_pixels[i], res_images[i] = self.getNumOfEqualPixels(countour_area_image, hsv_image_masks[i])
                f_name = self.features[i][0]
                verbose_text += "  | "+f_name+" no. of px ="+str(num_of_pixels[i]) + " |\n"

            # highest number of pixels equal to the hsv mask
            highest_num = max(num_of_pixels)

            # get the index of the feature
            feature_type = -1
            for i in range(0, self.num_of_feature_types):
                if(num_of_pixels[i] == highest_num and num_of_pixels[i] != 0):
                    feature_type = i

            # setup som print data
            print_text += "  Contour"+str(counter)+" centroid: ("+str(centroid_x)+","+str(centroid_y)+") "

            # some checks
            if(feature_type == -1):
                print(print_text + "unidentified")
                continue

            # bean data
            feature_name = self.features[feature_type][0]
            self.identifiedContours.append((feature_name, cnt))

            # draw contour
            self.drawIdentifiedContour(counter)

            # print text
            print_text += "identified Object with feature:"+feature_name
            print(print_text)

            # verbose display
            if(self.verbose):
                print(verbose_text)
                self.displayImage("resulting_countours_image", self.image_result.copy())
                self.displayImage("contour area image", countour_area_image)
                for i in range(0, self.num_of_feature_types):
                    self.displayImage("mask_"+self.features[i][0], hsv_image_masks[i])
                    if not (self.displayImage("result bitwise_and", res_images[i], wait=True, kill=True)):
                        cv2.destroyWindow("mask_"+self.features[i][0])
                        break
                    cv2.destroyWindow("mask_"+self.features[i][0])
                cv2.destroyWindow("resulting_countours_image")
                cv2.destroyWindow("contour area image")
            counter+=1
            continue

    def getNumOfEqualPixels(self, image1, image2, pixel_color=50):
        res_image = cv2.bitwise_and(image1,image2)
        points = np.where(res_image > pixel_color)
        num_of_pixels = len(points[0])
        return num_of_pixels, res_image

    # segmentation
    def getImageEdges(self, gray_image, dilation=0, lower=20, upper=50):
        # apply blur
        edges = cv2.GaussianBlur(gray_image.copy(),(27,27), 3)

        #canny
        edges = cv2.Canny(edges, lower, upper)

        # dilate
        image_edges = cv2.dilate(edges, self.kernel_3x3, iterations = dilation)
        return image_edges

    def fillWithinEdges(self, edges):
        # flood fill
        th, img_th = cv2.threshold(edges.copy(), 0, 1, cv2.THRESH_BINARY_INV);
        img_filled = img_th.copy()
        mask = np.zeros((self.height+2, self.width+2, 1), np.uint8)
        cv2.floodFill(img_filled, mask, (0,0), 255)
        return img_filled

    def getContourAreaImage(self, contour):
        # create new image
        contour_area_image = self.base_image_s.copy()

        # draw contour
        cv2.drawContours(contour_area_image, [contour], -1, (255,255,255), 3)

        # fill in hole
        contour_area_image = self.fillWithinEdges(contour_area_image)
        return contour_area_image

    def applyWaterShed(self):
        return

    # visualisation
    def displayImage(self, window_name, image, down_scale=True, wait=False, escape_key=ord('\x1b'),kill=False):
        rezised_image = image.copy()
        if(down_scale):
            rezised_image = cv2.resize(rezised_image, (self.display_width, self.display_height))
        cv2.imshow(window_name, rezised_image)

        if(wait):
            k = cv2.waitKey()
            if(kill):
                cv2.destroyWindow(window_name)
            if(k == escape_key):
                return False
        return True

    def writeTextOnImage(self, image, text, bottom_x, bottom_y, font_scale=1, font_rgb_color=(0,0,255), font=cv2.FONT_HERSHEY_SIMPLEX, lineType=2):
        # setup font
        bottomLeftCornerOfText = (bottom_x, bottom_y)

        #put text
        img = image.copy()
        cv2.putText(img, text,
        bottomLeftCornerOfText,
        font,
        font_scale,
        font_rgb_color,
        lineType)

        return img

    def drawIdentifiedContours(self):
        for i, tuple in enumerate(self.identifiedContours):
            self.drawIdentifiedContour(i)

    def drawIdentifiedContour(self,i):
        # get usefull data
        cnt = self.identifiedContours[i][1]
        M = cv2.moments(cnt)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])

        # draw contour
        cv2.circle(self.image_result,((int)(centroid_x), (int)(centroid_y)),10,(0,0,255),2)
        cv2.drawContours(self.image_result, [cnt], 0, (0,255,0), 3)
        self.image_result = self.writeTextOnImage(self.image_result, self.identifiedContours[i][0], centroid_x, centroid_y, font_rgb_color=(0,0,255))

#  main
if (__name__) == "__main__":

    # default vals
    camera = None
    converter = None

    # read args
    test_image = cv2.imread((os.getcwd()+'/images/image1.bmp'), -1)
    mode = [False, False, False, False] # mode[0] = single_view(False) or continous_view(True)  # mode[1] = view_image(False) or process image(True) # mode[2]  = view only(False) or add_feature(True)
    verbose = False
    for arg in sys.argv:
        if((arg == "h" or arg == "help") and len(sys.argv) == 2):
            print("running from ws:" + str(os.getcwd()))
            help_str = "default : will try to read image: /images/image1.bmp\n"
            help_str += "Options:\n\n"
            help_str += "with_image : (prompt) will ask for image in /images/\n"
            help_str += "with_camera : run contiounsly with camera\n"
            help_str += "with_images : read directory: /images/\n\n"
            help_str += "add_feature : add feature from hsv range to /features.json (loads on start)\n"
            help_str += "find_features: find features on given image using features defined in file /features.json\n\n"
            help_str += "verbose : print and display more images, for debugging\n"
            print(help_str)
            sys.exit()

        if(arg == "with_camera"):
            mode[0] = True  # run in continous mode with camera

        if(arg == "find_features"):
            mode[1] = True  # process images aswell

        if(arg == "add_feature"):
            mode[2] = True  # add feature using hsv image masks

        if(arg == "with_image"):
            print("test image name from images file:")
            img_pth = (os.getcwd()+'/images/'+input())
            if(os.path.exists(img_pth)):
                test_image = cv2.imread(img_pth, -1)
                image_queue.put(test_image)
            else:
                print("could not find image"+img_pth)
                print("exiting")
                sys.exit()

        if(arg == "with_images"):
            img_pth = (os.getcwd()+'/images/')
            if(os.path.isdir(img_pth)):
                image_paths = glob.glob(img_pth+"*.bmp")
                if(len(image_paths) > 0):
                    test_image = cv2.imread(image_paths[0], -1)
                    for pth in image_paths:
                        image_queue.put(cv2.imread(pth, -1))
                        print("loaded image:"+pth)
                else:
                    print("no image was found in "+ img_pth)
                    print("exiting")
                    sys.exit()
            else:
                print("directory "+ img_pth +" does not exist!")
                print("exiting")
                sys.exit()

        if(arg == "threading"):
            mode[2] = True # multi threading (will disable view from feature finder == no add_Feature and no display and always continous)

        if(arg == "verbose"):
            verbose = True

    # init camera or test image
    capture_thread = None
    display_q1_thread = None
    display_q2_thread = None
    camera = None
    converter = None

    # setup camera
    if(mode[0]):
        camera, converter = initCamera()
        capture_thread = threading.Thread(target=grabImagesThread, args=(camera,converter))
        capture_thread.start()

    # for threading
    if(mode[2]):
        display_q1_thread = threading.Thread(target=displayQueueThread, args=("result image",queue1))
        display_q2_thread = threading.Thread(target=displayQueueThread, args=("original image",queue2))
        display_q1_thread.start()
        display_q2_thread.start()

    # create bean finder
    bean_finder = FeatureFinder(mode, image_queue, display_queue_1=queue1, display_queue_2=queue2, verbose=verbose, height=test_image.shape[0], width=test_image.shape[1], name="bean finder")

    # run
    bean_finder.run()

    #exit
    exit_threads = True
    cv2.destroyAllWindows()
    if(camera):
        camera.StopGrabbing()
    if(display_q1_thread):
        display_q1_thread.join()
    if(display_q2_thread):
        display_q2_thread.join()
    sys.exit()
