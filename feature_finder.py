# author:
# Johan Mgina
# Date modified: 15-6-2019
# version: 1.0.0
# Find epllipse like objects on an image using hsv colors and canny edges image

import cv2
import time
import numpy as np
import re
import json
import os
from sequential_item_buffer import *


## find ellipse like objects using hsv colors
class FeatureFinder:
    # main routine functions
    def __init__(self, mode, in_image_buffer=SequentialItemBuffer("in_image_buffer"), result_image_buffer=SequentialItemBuffer("result_images"), original_image_buffer=SequentialItemBuffer("original_images"), height=1554, width=2074, name="feature_finder", verbose=False, max_contours=50, sleep_time_sec=0.100):
        # possibility to use lists and locks from outside
        self.public_self = self

        # setup
        self.stop = True
        self.name = name
        self.mode = mode
        self.in_image_buffer = in_image_buffer
        self.result_image_buffer = result_image_buffer
        self.original_image_buffer = original_image_buffer
        self.sleep_time = sleep_time_sec
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
            # single
            if not (self.mode[0]):
                self.delay = 0 # wait for input

            # contious
            if(self.mode[0]):
                self.delay = 50 # 1 ms

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
                self.delay = 16 # about 60 fps

            # pick_color mode when running camera
            if(self.mode[0] and self.mode[2]):
                self.delay = 250 # 4 fps

            # print input
            print_text += self.name+" input list:\n"
            if (self.mode[2]):
                print_text += "  : click on a pixel on window 'HSV_color_picker' to create a hsv image mask\n"
                print_text += "  : use <arrow keys left and right> to change hsv range Value for the mask\n"
                print_text += "  : use <arrow keys up and down> to change hsv ranges HUE and Saturation for the mask\n"
                print_text += "  : press <spacebar> to add ranges feature\n"

            if (self.mode[0] and not self.mode[1] and not self.mode[2]):
                print_text += "  : press <spacebar> to make image\n"

            print_text += "  : press <esc> to exit\n"

            if(self.delay == 0):
                print_text += "  : press any other key to continue to next image\n"

        verbose_text += self.name+": set delay to "+ str(self.delay) + "\n"
        if(self.verbose):
            print(verbose_text)


        print(self.name+": succesfully initialised\n" + print_text)

    def run(self):
        # for util
        self.stop = False
        tick_start = 0
        tick_end = 0

        # process images
        while not (self.stop):
            # get image
            is_new_image, seq_image = self.in_image_buffer.getFirst(timeout=33)

            if(is_new_image):
                # get image and set data
                seq_nr, image = seq_image
                self.current_image = image.copy()

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
                    # self.solution1()
                    self.solution2()

                    # print data
                    tick_end = cv2.getTickCount()
                    self.time_between_images = (tick_end - tick_start)/ cv2.getTickFrequency()
                    print(self.name+": took "+str(self.time_between_images)+" to process image")

                # add to out buffers
                if (self.mode[3]):
                    if (self.result_image_buffer.putLast(seq_image=(seq_nr, self.image_result.copy()))):
                        print(self.name+": appended img result "+str(seq_nr)+" to result buffer")
                    if(self.original_image_buffer.putLast(seq_image=(seq_nr, self.current_image.copy()))):
                        print(self.name+": appended img origin "+str(seq_nr)+" to origin buffer")
                else:
                    if(self.mode[3]):
                        time.sleep(self.sleep_time) # give other threads some time

            # display images and handle input
            if (not self.mode[3]):
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

        # exit window for mode[3]
        # if(self.mode[3]):
        #    img_exit = np.zeros((75,75,3), np.uint8)
        #    img_exit = self.writeTextOnImage(img_exit, "esc",20, 50,font_scale=1)
        #    self.displayImage(self.name+"_exit_window", img_exit, down_scale=False)

        # repeat for next keys
        while(key == -1):
            key = -2
            # delay get get input
            key = cv2.waitKey((int)(self.delay))

            # if can input
            if not (self.mode[3]):

                # spacebar
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
                if(key == 82):
                    if(self.mode[2]):
                        self.color_picker_range[0] += 2
                        self.color_picker_range[1] += 2
                        self.updateColorPickerDisplay()
                        print(self.name+": new hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))
                        key = -1

                # arror key down
                if(key == 84):
                    if(self.mode[2]):
                        self.color_picker_range[0] -= 2
                        self.color_picker_range[1] -= 2
                        self.updateColorPickerDisplay()
                        print(self.name+": hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))
                        key = -1

                # arror key right
                if(key == 83):
                    if(self.mode[2]):
                        self.color_picker_range[2] += 5
                        self.updateColorPickerDisplay()
                        print(self.name+": hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))
                        key = -1

                # arror key left
                if(key == 81):
                    if(self.mode[2]):
                        self.color_picker_range[2] -= 5
                        self.updateColorPickerDisplay()
                        print(self.name+": hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))
                        key = -1

            # escape
            if (key == ord('\x1b')):
                return False
        return True

    def displayMaskFromChosenPixel(self, event, x, y, f, p):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_pixel = self.color_picker_image[y,x]
            self.updateColorPickerDisplay()
            print(self.name+": hsv range for masking "+str(self.color_picker_range_low) +"-"+str(self.color_picker_range_high))

    # solutions
    def solution1(self, use_foreground=False):
        # enhancement
        img_b_gray = cv2.split(self.current_image.copy())[0]

        # edges using canny
        new_edges_image = self.getImageEdges(img_b_gray, dilation=3, lower=20, upper=50)
        old_edges_image = self.image_edges.copy()

        # get difference
        diff_value = -1
        diff_image = self.base_image_s.copy()
        if(old_edges_image.shape[0] == self.height and old_edges_image.shape[1] == self.width):
            diff_image = cv2.bitwise_xor(self.image_edges.copy(), new_edges_image.copy())
            diff_value = len((np.where(diff_image == 255))[0])


        # identify if difference is high
        if(diff_value > 50000 or diff_value == -1 or len(self.identifiedContours) == 0):
            # new image
            self.image_edges = new_edges_image.copy()

            # fill in holes
            self.image_filled = self.fillWithinEdges(self.image_edges.copy())

            # make sure foreground image
            if(use_foreground == True):
                self.image_filled = self.getForegroundImage(self.image_filled.copy())

            # display extra text and display
            if(verbose):
                print(self.name+": difference value between image:"+str(diff_value))
                self.displayImage("image_edges", self.image_edges)
                self.displayImage("image_filled", self.image_filled, wait=True, kill=True)
                cv2.destroyWindow("image_edges")

            # mark beans
            self.identifyFilledImageUsingHsvRanges()

        else:
            if(verbose):
                print(self.name+": difference between images to small!! value: "+str(diff_value))
                print(self.name+": redrawing contours..")

            # redraw if difference is low
            self.drawIdentifiedContours()

    def solution2(self):
        # gray scale image
        img = self.current_image.copy()
        img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

        # equalization
        img_gray_equ = self.equalizeHistogram(img_gray)

        # threshold image
        img_thresh = self.getMedianThresholdImage(img_gray_equ)

        # get foreground image
        img_foreground = self.getForegroundImage(img_thresh, invert=False)


        # get background image
        img_background = cv2.dilate(img_thresh, self.kernel_3x3, iterations=1)

        # substract (edges, kinda)
        img_in_between = img_background - img_foreground

        self.displayImage("img_thresh", img_thresh)
        self.displayImage("img_between", img_in_between)
        self.displayImage("img_foreground", img_foreground)
        self.displayImage("img_background", img_background, wait=True)



        # apply watershed
        # read markers

        # set result
        self.image_result = self.current_image.copy()


    # feature extraction
    def identifyFilledImageUsingHsvRanges(self, sensitivity=0.00025):
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

        # to print
        print_text = ""
        verbose_text = ""

        # for each contour and feature check if within its hsv range
        for cnt in contours:
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
            min_num_pixels = (int)((self.height*self.width)* sensitivity)
            if(highest_num >= min_num_pixels):
                for i in range(0, self.num_of_feature_types):
                    if(num_of_pixels[i] == highest_num):
                        feature_type = i

            # double checks
            if(feature_type == -1):
                verbose_text += " - Contour skipped, is lower than minimum num pixels:" + str(min_num_pixels) + "\n\n"
                continue

            # bean data
            feature_name = self.features[feature_type][0]
            self.identifiedContours.append((feature_name, cnt))

            # draw contour
            self.drawIdentifiedContour(counter)

            # setup some print data
            txt = "  - Contour"+str(counter)+" center:("+str(centroid_x)+","+str(centroid_y)+") Hsv Feature:"+feature_name+"\n"
            print_text += txt
            verbose_text += txt

            # verbose display
            if(self.verbose):
                print(verbose_text)
                combined_image_1 = np.vstack((cv2.cvtColor(countour_area_image.copy(), cv2.COLOR_GRAY2BGR), self.image_result.copy()))
                for i in range(0, self.num_of_feature_types):
                    combined_image_2 = np.vstack((hsv_image_masks[i].copy(), res_images[i].copy()))
                    if not (self.displayImage("result bitwise_and", np.hstack((cv2.cvtColor(combined_image_2.copy(), cv2.COLOR_GRAY2BGR), combined_image_1.copy())), wait=True, kill=True)):
                        break
                cv2.destroyWindow("resulting_countours_image")
                cv2.destroyWindow("contour area image")
            counter+=1
            continue
        if not (verbose):
            print(print_text)
        else:
            print(verbose_text)

    def getNumOfEqualPixels(self, image1, image2, pixel_color=50):
        res_image = cv2.bitwise_and(image1,image2)
        points = np.where(res_image > pixel_color)
        num_of_pixels = len(points[0])
        return num_of_pixels, res_image

    # segmentation
    def getImageEdges(self, gray_image, dilation=0, lower=20, upper=40):
        # apply blur
        edges = cv2.GaussianBlur(gray_image.copy(),(27,27), 1) # instead of 1 to 3 if removing bilateral filter
        edges_bi = cv2.bilateralFilter(edges,9,75,75)

        #canny
        edges_can = cv2.Canny(edges_bi, lower, upper)

        # dilate
        image_edges = cv2.dilate(edges_can, self.kernel_3x3, iterations = dilation)

        if(verbose):
            cv2.destroyAllWindows()
            self.displayImage("gausblur", edges)
            self.displayImage("after bilateral filter", edges_bi)
            self.displayImage("canny result", image_edges, wait=True)

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

    def getMedianThresholdImage(self, img_gray):
        hist,bins = np.histogram(img_gray.ravel(),256,[0,256])
        import statistics
        median = statistics.median(hist)

        marking_thresh = 0
        for current in range(0, len(hist)):
            if(hist[current] > median-100 and hist[current] < median+100):
                if(abs(hist[current]-median) <= abs(hist[marking_thresh]-median)):
                    marking_thresh = current

        out_image = cv2.inRange(img_gray, 0, marking_thresh)
        return out_image

    def getForegroundImage(self, img_filled, invert=True):
        img_in = img_filled.copy()

        # invert
        if(invert == True):
            img_in = cv2.bitwise_not(img_in)

        # Finding sure foreground area
        dist_tf = cv2.distanceTransform(img_in,cv2.DIST_L2,5)
        ret, foreground = cv2.threshold(dist_tf,0.2*dist_tf.max(),255,0)

        out_image = self.base_image_s.copy()
        if(invert):
            out_image[foreground == 0] = 255  # invert again
        else:
            out_image = foreground.copy()
        return out_image

    # enhancement
    def equalizeHistogram(self, gray_image, tile_grid_size_x=8, tile_grid_size_y=8):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_grid_size_x,tile_grid_size_y))
        equ_image = clahe.apply(gray_image)
        return equ_image

    # visualisation
    def displayImage(self, window_name, image, down_scale=True, wait=False, escape_key=ord('\x1b'),kill=False):
        rezised_image = image.copy()
        h = self.display_height
        w = self.display_width

        if(down_scale):
            rezised_image = cv2.resize(rezised_image, (w, h))
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
