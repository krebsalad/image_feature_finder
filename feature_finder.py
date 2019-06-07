# author:
# Johan Mgina
# Date modified: 6-6-2019
# version: 0.9.5
# Todo next: watershed for overlapping beans, relative directories and also arguments

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


## image sharing ##
image_queue = queue.Queue()
q_siz = 3
camera_w = 2074
camera_h = 1554

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

def grabImages(camera, converter):
    while camera.IsGrabbing():
        # grab image when queue is not full
        if(image_queue.qsize() < q_size):
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img = image.GetArray()
                image_queue.put(img)
            grabResult.Release()

    print("camera stopped!!")


## find ellipse like objects using hsv colors ##
class FeatureFinder:
    # main routine functions
    def __init__(self, mode, image_queue, height=1554, width=2074, capture_thread=None, name="feature_finder", verbose=False):
        # setup
        self.name = name
        self.mode = mode
        self.image_queue = image_queue
        self.capture_thread = capture_thread
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
        self.delay = 0
        self.time_between_images = 0
        self.down_scale_for_display = 3
        self.snapshot_counter = 0

        # create new file if not available
        if not (os.path.exists(self.save_file)):
            with open(self.save_file, "w+") as file:
                file.write("[]")

        # load data
        with open(self.save_file) as file:
                self.features = json.load(file)

        self.num_of_feature_types = len(self.features)

        # const images
        self.base_image = np.zeros((self.height,self.width,3), np.uint8)
        self.base_image_s = np.zeros((self.height,self.width,1), np.uint8)

        # changing images
        self.current_image = self.base_image.copy()

        self.image_hsv = cv2.cvtColor(self.base_image.copy(),cv2.COLOR_BGR2HSV)
        self.image_result = self.base_image.copy()

        self.image_edges = self.base_image_s.copy()
        self.image_filled = self.base_image_s.copy()

        # color picker vals
        self.clicked_bean_low = [0,0,0]
        self.clicked_bean_high = [0,0,0]

        # kernels
        self.kernel_3x3 = np.array(([-1, -1, -1],[-1, 8, -1],[-1, -1, -1]))


        # capture and run requires camera
        if(self.mode[0]):
            self.delay = 1 # 1 ms
            self.capture_thread.start()
            print("camera is running..")

        # pick a color from a hsv image and create mask
        if(self.mode[1]):
            # create mask
            self.color_picker_image = cv2.resize(self.image_hsv.copy(), ((int)(self.width/self.down_scale_for_display)+1, (int)(self.height/self.down_scale_for_display)+1))
            self.color_picker_mask = cv2.resize(self.base_image_s.copy(), ((int)(self.width/self.down_scale_for_display)+1, (int)(self.height/self.down_scale_for_display)+1))

            # display images
            self.displayImage("Mask Result", self.color_picker_mask, down_scale=1)
            self.displayImage("HSV_color_picker", self.color_picker_image, down_scale=1)

            # callback on click
            cv2.setMouseCallback("HSV_color_picker", self.displayMaskFromChosenPixel)

            # info
            print("click on a pixel on window 'HSV_color_picker' to create a hsv image mask")
            print("press <spacebar> on to add ranges as a feature")

        # specific modes
        if(self.mode[0] and not self.mode[1]):
            print("press <spacebar> to take image")

        if(self.mode[0] and self.mode[1]):
            self.delay = 1000 # 1 s

        print("succesfully initialised feature finder: "+self.name)

    def run(self):
        # for util
        tick_start = 0
        tick_end = 0

        # process images
        while(True):
            if not (self.image_queue.empty()):
                print("processing image")
                # get image
                self.current_image = self.image_queue.get(block=True, timeout=None)

                # timer
                tick_start = cv2.getTickCount()

                # process image
                if not (self.mode[1]):
                    self.solution1()
                    self.displayImage("result", self.image_result)

                if(self.mode[1]):
                    self.image_hsv = cv2.cvtColor(self.current_image.copy(),cv2.COLOR_BGR2HSV)
                    self.color_picker_image = cv2.resize(self.image_hsv.copy(), ((int)(self.width/self.down_scale_for_display)+1, (int)(self.height/self.down_scale_for_display)+1))
                    self.displayImage("HSV_color_picker", self.color_picker_image, down_scale=1)

                tick_end = cv2.getTickCount()
                self.time_between_images = (tick_end - tick_start)/ cv2.getTickFrequency()
                print("took "+str(self.time_between_images)+" to process image")

            # original image
            self.displayImage("original", self.current_image)
            if not (self.handleInput()):
                print("exiting..")
                return

    # input
    def handleInput(self):
        # delay get get input
        key = cv2.waitKey((int)(self.delay))

        # spacebar
        if (key == 32):
            # add bean
            if(self.mode[1]):
                # get input
                print("Name of to identify feature:")
                in_str = str(input())

                # text
                new_bean = [in_str, self.clicked_bean_low, self.clicked_bean_high]
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

                print("saved features at "+self.save_file)

            # make snapshot
            else:
                self.snapshot_counter +=1
                pth = self.images_dir+"/image"+ str(self.snapshot_counter) + ".bmp"
                if(os.path.exists(self.images_dir+"/")):
                    cv2.imwrite(pth,self.current_image)
                    print("image made and saved in: " + pth)
                else:
                    print("could not find dir: "+ self.images_dir)

        # escape
        if (key == ord('\x1b')):
            return False
        return True

    def displayMaskFromChosenPixel(self, event, x, y, f, p):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = self.color_picker_image[y,x]
            self.clicked_bean_high = [pixel[0] + 10, pixel[1] + 10, pixel[2] + 40]
            self.clicked_bean_low = [pixel[0] - 10, pixel[1] - 10, pixel[2] - 40]

            self.color_picker_mask = cv2.inRange(self.color_picker_image,np.array(self.clicked_bean_low),np.array(self.clicked_bean_high))
            self.displayImage("Mask Result", self.color_picker_mask, down_scale=1)

            print("picked color:", pixel)
            print("low-high color ranges:", self.clicked_bean_high, self.clicked_bean_low)

    # solutions
    def solution1(self):
        # enhancement
        img_b_gray = cv2.split(self.current_image)[0]

        # edges
        new_edges_image = self.getImageEdges(img_b_gray, dilation=2, lower=20, upper=50)

        # get difference
        diff = cv2.bitwise_xor(self.image_edges.copy(), new_edges_image.copy())
        diff_value = len((np.where(diff == 255))[0])

        # identify if difference is high
        if(diff_value > 50000):
            self.image_edges = new_edges_image
            # fill in holes
            self.image_filled = self.fillWithinEdges(self.image_edges)

            # mark beans
            self.identifyFilledImageUsingHsvRanges()
        # redraw if difference is low
        else:
            self.drawIdentifiedContours()
            if(self.verbose):
                print("Skipping image processing..., Difference value = "+str(diff_value))
                self.displayImage("image_edges", self.image_edges)
                self.displayImage("new_edges_image", new_edges_image)
                self.displayImage("diff", diff, wait=True)

    # feature extraction
    def identifyFilledImageUsingHsvRanges(self):
        self.identifiedContours.clear()
        self.num_of_feature_types = len(self.features)

        # are there feature available
        if(self.num_of_feature_types == 0):
            print("No features available!!!")
            self.result_image = self.base_image.copy()
            return

        # hsv image
        self.image_hsv = cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGR2HSV)

        # create bean masks
        hsv_image_masks = [None]*self.num_of_feature_types
        for i in range(0, self.num_of_feature_types):
            mask = cv2.inRange(self.image_hsv, np.array(self.features[i][1]), np.array(self.features[i][2]))
            hsv_image_masks[i] = mask

        # find contours
        ret, thresh = cv2.threshold(self.image_filled, 100, 200, 3)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # identify contours
        self.image_result = self.current_image.copy()
        counter = 0
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
            verbose_text += "num of pixels on ranges:\n"
            for i in range(0, self.num_of_feature_types):
                num_of_pixels[i], res_images[i] = self.getNumOfEqualPixels(countour_area_image, hsv_image_masks[i])
                f_name = self.features[i][0]
                verbose_text += "| nr px "+f_name+"="+str(num_of_pixels[i]) + " |"

            # highest number of pixels equal to the hsv mask
            highest_num = max(num_of_pixels)

            # get the index of the feature
            feature_type = -1
            for i in range(0, self.num_of_feature_types):
                if(num_of_pixels[i] == highest_num and num_of_pixels[i] != 0):
                    feature_type = i

            # setup som print data
            print_text = "Contour"+str(counter)+" centroid: ("+str(centroid_x)+","+str(centroid_y)+") "

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
                self.displayImage("contour are image", countour_area_image)
                for i in range(0, self.num_of_feature_types):
                    self.displayImage("mask_"+self.features[i][0], hsv_image_masks[i])
                    self.displayImage("result bitwise_and", res_images[i], wait=True)
                    cv2.destroyWindow("mask_"+self.features[i][0])
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
        edges = cv2.GaussianBlur(gray_image,(27,27), 3)

        #canny
        edges = cv2.Canny(edges, lower, upper)

        # dilate
        image_edges = cv2.dilate(edges, self.kernel_3x3, iterations = dilation)
        return image_edges

    def fillWithinEdges(self, edges):
        # flood fill
        th, img_th = cv2.threshold(edges, 0, 1, cv2.THRESH_BINARY_INV);
        img_filled = img_th.copy()
        mask = np.zeros((self.height+2, self.width+2), np.uint8)
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
    def displayImage(self, window_name, image, down_scale=3, wait=False, kill=False):
        height, width = image.shape[:2]
        rezised_image = image.copy()
        rezised_image = cv2.resize(image, ((int)(width/down_scale)+1, (int)(height/down_scale)+1))
        cv2.imshow(window_name, rezised_image)
        if(wait):
            cv2.waitKey()
        if(kill):
            cv2.destroyWindow(window_name)

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
    print("running from ws:" + str(os.getcwd()))

    # default vals
    camera = None
    converter = None
    delay = 1
    test_image = cv2.imread((os.getcwd()+'/images/Bonen_110k_diaf_8_gain_0.bmp'), -1)


    # read args
    mode = [False, False] # mode[0] = process test image(False) or from cam(True)  # mode[1] = alternativly view only(False) or add_feature(True)
    for arg in sys.argv:
        if((arg == "h" or arg == "help") and len(sys.argv) == 2):
            print("\nOptions:\nrun_camera : run with camera\nadd_feature : add feature from hsv range to /beans.json (loads on start)\nimage : start with image from /images/ directory")
            sys.exit()

        if(arg == "run_camera"):
            mode[0] = True

        if(arg == "add_feature"):
            mode[1] = True

        if(arg == "image"):
            print("test image name from images file:")
            test_image = cv2.imread((os.getcwd()+'/images/'+input()), -1)

    # init camera or test image
    capture_thread = None
    camera = None
    converter = None
    if(mode[0]):
        camera, converter = initCamera()
        capture_thread = threading.Thread(target=grabImages, args=(camera,converter))
    else:
        image_queue.put(test_image)

    # create bean finder
    bean_finder = FeatureFinder(mode, image_queue, verbose=False, height=test_image.shape[0], width=test_image.shape[1], capture_thread=capture_thread, name="bean finder")

    # run
    bean_finder.run()

    cv2.destroyAllWindows()
    if(camera):
        camera.StopGrabbing()
    sys.exit()
