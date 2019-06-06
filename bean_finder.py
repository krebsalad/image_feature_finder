# author:
# Johan Mgina
# Date modified: 6-6-2019
# version: 0.9.1
# Todo next: watershed for overlapping beans, relative directories and also arguments

from pypylon import pylon
import cv2
import time
import sys
import numpy as np
from matplotlib import pyplot as plt
import threading
import queue
import re
import json
import os


# image vals
image_queue = queue.Queue()
q_size = 3
time_between_images = 0
camera_w = 2074
camera_h = 1554

# color picker vals
color_picker_image = None
clicked_bean_low = [0,0,0]
clicked_bean_high = [0,0,0]

# color ranges of the beans
features = []

#paths
current_dir = os.getcwd()
images_dir = current_dir + "/images"

# kernels
kernel_3x3 = np.array(([-1, -1, -1],[-1, 8, -1],[-1, -1, -1]))


# image grabbing
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
        else:
            # wait until atleast one image is processed
            if((time_between_images) > 0):
                time.sleep(time_between_images)

    print("camera stopped!!")

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


# display images
def displayImage(window_name, image, down_scale=3, wait=False, kill=False):
    height, width = image.shape[:2]
    rezised_image = image.copy()
    rezised_image = cv2.resize(image, ((int)(width/down_scale)+1, (int)(height/down_scale)+1))
    cv2.imshow(window_name, rezised_image)
    if(wait):
        cv2.waitKey()
    if(kill):
        cv2.destroyWindow(window_name)

def displayImageHistogram(image):
    # plot histogram
    plt.hist(image.ravel(),256,[0,256])
    plt.show()

def displayMaskFromChosenPixel(event,x,y,f,p):
    global clicked_bean_high
    global clicked_bean_low
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = color_picker_image[y,x]
        clicked_bean_high = [pixel[0] + 10, pixel[1] + 10, pixel[2] + 40]
        clicked_bean_low = [pixel[0] - 10, pixel[1] - 10, pixel[2] - 40]

        image_mask = cv2.inRange(color_picker_image,np.array(clicked_bean_low),np.array(clicked_bean_high))
        displayImage("Mask Result", image_mask, down_scale=1)

        print("picked color:", pixel)
        print("low-high color ranges:", clicked_bean_high, clicked_bean_low)


# image transformation (functions return an edited image)
def equalizeHistogram(gray_image, tile_grid_size_x, tile_grid_size_y):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ_image = clahe.apply(gray_image)
    return equ_image

def fillInHoles(img):
    # flood fill
    th, img_th = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV);
    img_filled = img_th.copy()
    h, w = img_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(img_filled, mask, (0,0), 255)
    return img_filled

def getImageEdges(gray_image, dilation=0, lower=20, upper=50):
    # apply blur
    edges = cv2.GaussianBlur(gray_image,(27,27), 3)
    # edges = cv2.bilateralFilter(edges,9,75,75)

    #canny
    edges = cv2.Canny(edges, lower, upper)

    # dilate
    edges = cv2.dilate(edges, kernel_3x3, iterations = dilation)
    return edges

def getBgrGrayscaledImages(image):
    # returns im_gray, im_b_gray, im_g_gray, im_r_gray
    b, g, r = cv2.split(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), b, g, r

def markPixelsAboveMedian(gray_image):
    marking_thresh = getMedianBin(gray_image)
    out_image = cv2.inRange(gray_image, 0, marking_thresh)
    return out_image

def writeTextOnImage(image, text, bottom_x, bottom_y, font_scale=1, font_rgb_color=(0,0,255), font=cv2.FONT_HERSHEY_SIMPLEX, lineType=2):
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

def closeImageEdges(image_edges, morph_kernel=None):
    # set kernel when none was set
    if not (morph_kernel):
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))

    # close gaps
    out_image = cv2.morphologyEx(image_edges.copy(), cv2.MORPH_CLOSE, morph_kernel)
    return out_image

def getContourAreaImage(contour, img_height, img_width):
    # create new image
    contour_area_image = np.zeros((img_height, img_width), np.uint8)

    # draw contour
    cv2.drawContours(contour_area_image, [contour], -1, (255,255,255), 3)

    # fill in hole
    contour_area_image = fillInHoles(contour_area_image)
    return contour_area_image


# util / math
def isHsvColorInRange(pixel_color, color_low, color_high):
    # lows and highs
    if(pixel_color[0] >= color_low[0] and pixel_color[0] <= color_high[0]):
        if(pixel_color[1] >= color_low[1] and pixel_color[1] <= color_high[1]):
            if(pixel_color[2] >= color_low[2] and pixel_color[2] <= color_high[2]):
                return True

    return False

def getMedianBin(img_gray):
    hist,bins = np.histogram(img_gray.ravel(),256,[0,256])
    import statistics
    median = statistics.median(hist)

    last = 0
    for current in range(0, len(hist)):
        if(hist[current] > median-100 and hist[current] < median+100):
            if(abs(hist[current]-median) <= abs(hist[last]-median)):
                last = current

    return last

def getAverageColorOfContour(img_hsv, contour):
    # get sizes
    height, width = img_hsv.shape[:2]

    # get points where ellipse if found and get color from hsv
    colors = []
    pts = getCoordsOfEllipseContour(contour, height, width)
    size = len(pts[0])
    for i in range(0, size):
        colors.append(img_hsv[pts[0][i], pts[1][i]])

    # calculate average color
    color_sum = [0,0,0]

    for color in colors:
        color_sum += color

    color_num = len(colors)

    if(color_num == 0):
        return [0,0,0]

    color_average = color_sum/color_num
    return color_average

def getCoordsOfEllipseContour(contour, img_height, img_width):
    #get area image
    contour_area_image = getContourAreaImage(contour, img_height, img_width)

    # get points where image is black
    pts = np.where(contour_area_image == 0)
    return pts

def getNumOfEqualPixels(image1, image2, pixel_color=50):
    res_image = cv2.bitwise_and(image1,image2)
    # displayImage("res", res, wait=True)
    points = np.where(res_image > pixel_color)
    num_of_pixels = len(points[0])
    return num_of_pixels, res_image

# bean selection
def markBeanContoursOnImage(img_filled, img_original, find_color_alg="using_mask", show_result=False, show_num=3):
    # read image
    height, width = img_original.shape[:2]

    # create hsv image
    img_hsv = cv2.cvtColor(img_original.copy(), cv2.COLOR_BGR2HSV)

    # create mask
    image_mask_kidney = None
    image_mask_zwarte = None
    image_mask_bruine = None
    image_mask_koffie = None
    if(find_color_alg == "using_mask"):
        image_mask_kidney = cv2.inRange(img_hsv, np.array(features[0][1]),np.array(features[0][2]))
        image_mask_zwarte = cv2.inRange(img_hsv, np.array(features[3][1]),np.array(features[3][2]))
        image_mask_bruine = cv2.inRange(img_hsv, np.array(features[2][1]),np.array(features[2][2]))
        image_mask_koffie = cv2.inRange(img_hsv, np.array(features[1][1]),np.array(features[1][2]))

    # find contours
    ret, thresh = cv2.threshold(img_filled, 100, 200, 3)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # identify contours
    img_contours = img_original.copy()
    counter = 0
    for cnt in contours:
        # filter contours
        if((int)(len(cnt)) < 30):
            continue

        # draw contour
        cv2.drawContours(img_contours, [cnt], 0, (0,255,0), 3)

        # get contour centroid
        M = cv2.moments(cnt)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])

        # draw circle on centroid
        cv2.circle(img_contours,((int)(centroid_x), (int)(centroid_y)),10,(0,0,255),2)
        counter+=1
        print_text = "Contour"+str(counter)+" centroid: ("+str(centroid_x)+","+str(centroid_y)+") color: "

        # check if centroids within bounds
        if(centroid_x < width and centroid_y < height):

            # indentify bean based on color
            if(find_color_alg=="centroid" or find_color_alg=="contour_average"):
                # set color of found bean
                if(find_color_alg=="centroid"):
                    pixel_color = img_hsv[centroid_y, centroid_x]

                if(find_color_alg=="contour_average"):
                    pixel_color = getAverageColorOfContour(img_hsv, cnt)

                print_text += str(pixel_color)

                # kindney boon
                if(isHsvColorInRange(pixel_color, features[0][1], features[0][2])):
                    img_contours = writeTextOnImage(img_contours, "kidney boon", centroid_x, centroid_y)
                    print_text += " - Type: kidney boon "

                # koffie bonen
                if(isHsvColorInRange(pixel_color, features[1][1], features[1][2])):
                    img_contours = writeTextOnImage(img_contours, "koffie boon", centroid_x, centroid_y)
                    print_text += " - Type: koffie boon "

                # bruine boon
                if(isHsvColorInRange(pixel_color, features[2][1], features[2][2])):
                    img_contours = writeTextOnImage(img_contours, "bruine boon", centroid_x, centroid_y)
                    print_text += " - Type: bruine boon "

                # zwarte boon
                if(isHsvColorInRange(pixel_color, bfeatures[3][1], features[3][2])):
                    img_contours = writeTextOnImage(img_contours, "zwarte boon", centroid_x, centroid_y)
                    print_text += " - Type: zwarte boon "

                print(print_text)
                if(show_result and counter < show_num):
                    temp = img_hsv.copy()
                    if(find_color_alg=="centroid"):
                        cv2.circle(temp,((int)(centroid_x), (int)(centroid_y)),10,(255,255,255),2)
                    if(find_color_alg=="contour_average"):
                        cv2.drawContours(temp, [cnt], 0, (0,255,0), 3)
                    displayImage("image_hsv", temp)
                    displayImage("img_contours", img_contours, wait=True)

            # identify bean based on hsv image mask
            if(find_color_alg == "using_mask"):
                # get number of pixel of each color this bean has
                countour_area_image = getContourAreaImage(cnt, height, width)
                countour_area_image = cv2.bitwise_not(countour_area_image)

                # num of pixels for each mask
                bru_num, res1 = getNumOfEqualPixels(countour_area_image, image_mask_bruine)
                kof_num, res2 = getNumOfEqualPixels(countour_area_image, image_mask_koffie)
                zwa_num, res3 = getNumOfEqualPixels(countour_area_image, image_mask_zwarte)
                kid_num, res4 = getNumOfEqualPixels(countour_area_image, image_mask_kidney)

                # highest number of pixels equal to the hsv mask
                highest_num = max([bru_num, kof_num, zwa_num, kid_num])

                # color for print
                pixel_color = img_hsv[centroid_y, centroid_x]
                print_text += str(pixel_color)

                if(bru_num == highest_num and kof_num != highest_num and zwa_num != highest_num and kid_num != highest_num):
                    img_contours = writeTextOnImage(img_contours, "bruine boon", centroid_x, centroid_y, font_rgb_color=(0,0,255))
                    print_text += " - Type: bruine boon "

                if(kof_num == highest_num and bru_num != highest_num and zwa_num != highest_num and kid_num != highest_num):
                    img_contours = writeTextOnImage(img_contours, "koffie boon", centroid_x, centroid_y, font_rgb_color=(0,255,0))
                    print_text += " - Type: koffie boon "

                if(zwa_num == highest_num and kof_num != highest_num and bru_num != highest_num and kid_num != highest_num):
                    img_contours = writeTextOnImage(img_contours, "zwarte boon", centroid_x, centroid_y, font_rgb_color=(255,0,0))
                    print_text += " - Type: zwarte boon "

                if(kid_num == highest_num and kof_num != highest_num and zwa_num != highest_num and bru_num != highest_num):
                    img_contours = writeTextOnImage(img_contours, "kidney boon", centroid_x, centroid_y, font_rgb_color=(0,255,0))
                    print_text += " - Type: kidney boon "

                print(print_text)
                print("masked pixel counts: bruine: "+str(bru_num)+" koffie: "+str(kof_num)+" zwarte: "+str(zwa_num)+" kidney: "+str(kid_num))

                if(show_result and counter < show_num):
                    displayImage("contour area", countour_area_image)

                    displayImage("result bitwise_and", res1)
                    displayImage("mask bruine boon", image_mask_bruine, wait=True, kill=True)

                    displayImage("result bitwise_and", res2)
                    displayImage("mask koffie boon", image_mask_koffie, wait=True, kill=True)

                    displayImage("result bitwise_and", res3)
                    displayImage("mask zwarte boon", image_mask_zwarte, wait=True, kill=True)

                    displayImage("result bitwise_and", res4)
                    displayImage("mask kidney boon", image_mask_kidney, wait=True, kill=True)
                    cv2.destroyAllWindows()

        if(show_result):
            displayImage("img_contours", img_contours, wait=True)

    return img_contours

def identifyObjectsUsingHsvRanges(img_filled, img_original):
    # read image
    height, width = img_original.shape[:2]
    num_of_feature_types = len(features)

    # are there feature available
    if(num_of_feature_types == 0):
        print("No features available!!!")
        return np.zeros((height,width,1), np.uint8)

    # hsv image
    img_hsv = cv2.cvtColor(img_original.copy(), cv2.COLOR_BGR2HSV)

    # create bean masks
    hsv_image_masks = [None]*num_of_feature_types
    for i in range(0, num_of_feature_types):
        mask = cv2.inRange(img_hsv, np.array(features[i][1]), np.array(features[i][2]))
        hsv_image_masks[i] = mask
        # displayImage("mask_"+features[i][0], mask)

    # find contours
    ret, thresh = cv2.threshold(img_filled, 100, 200, 3)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # identify contours
    img_contours = img_original.copy()
    counter = 0
    for cnt in contours:
        # get usefull data
        M = cv2.moments(cnt)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])

        # filter contours
        if((int)(len(cnt)) < 30):
            continue

        if(centroid_x > width and centroid_y > height):
            continue

        # redraw contour in a threshold image
        countour_area_image = getContourAreaImage(cnt, height, width)
        countour_area_image = cv2.bitwise_not(countour_area_image)


        # get num of pixels for each mask
        num_of_pixels = [None]*num_of_feature_types
        for i in range(0, num_of_feature_types):
            num_of_pixels[i], res_image = getNumOfEqualPixels(countour_area_image, hsv_image_masks[i])

        # highest number of pixels equal to the hsv mask
        highest_num = max(num_of_pixels)

        # get the index of the feature
        feature_type = -1
        for i in range(0, num_of_feature_types):
            if(num_of_pixels[i] == highest_num and num_of_pixels[i] != 0):
                feature_type = i

        # setup som print data
        print_text = "Contour"+str(counter)+" centroid: ("+str(centroid_x)+","+str(centroid_y)+") "

        # some checks
        if(feature_type == -1):
            cv2.drawContours(img_contours, [cnt], 0, (0,255,0), 3)
            print(print_text + "unidentified")
            counter+=1
            continue

        # bean data
        feature_name = features[feature_type][0]

        # draw contour
        cv2.circle(img_contours,((int)(centroid_x), (int)(centroid_y)),10,(0,0,255),2)
        cv2.drawContours(img_contours, [cnt], 0, (0,255,0), 3)
        img_contours = writeTextOnImage(img_contours, feature_name, centroid_x, centroid_y, font_rgb_color=(0,0,255))

        # print text
        print(print_text + "identified Object with feature:"+feature_name)
        counter+=1
        continue

    return img_contours


# solutions
def solution1(image):
    # create grayscaled image
    img_gray, img_b_gray, img_g_gray, img_r_gray = getBgrGrayscaledImages(image)

    # edges
    img_edges = getImageEdges(img_b_gray, dilation=2, lower=20, upper=50)

    # fill in holes
    img_filled = fillInHoles(img_edges)

    # mark beans
    img_marked = identifyObjectsUsingHsvRanges(img_filled, image)

    return img_marked

def solution2(image):
    # create grayscaled image
    img_gray, img_b_gray, img_g_gray, img_r_gray = getBgrGrayscaledImages(image)

    # edges
    img_median = markPixelsAboveMedian(img_gray)

    # fillInHoles
    img_filled = fillInHoles(img_median)

    # mark beans
    img_marked = markBeanContoursOnImage(img_filled, image, find_color_alg="using_mask")

    return img_marked

def step_by_step_demo(image):
    # create grayscaled image
    img_gray, img_b_gray, img_g_gray, img_r_gray = getBgrGrayscaledImages(image)

    # histogram comparison
    displayImage("img_gray", img_gray, wait=True, kill=True)
    displayImageHistogram(img_gray)
    displayImage("img_b_gray", img_b_gray, wait=True, kill=True)
    displayImageHistogram(img_b_gray)
    displayImage("img_g_gray", img_g_gray, wait=True, kill=True)   # voor blauwe range gezien de kleuren van de bonen ver van de achtergrond af zitten richting zwart
    displayImageHistogram(img_g_gray)
    displayImage("img_r_gray", img_r_gray, wait=True, kill=True)
    displayImageHistogram(img_r_gray)

    # image edges
    img_edges_1 = getImageEdges(img_b_gray, dilation=4, lower=20, upper=50) # door het kiezen van een range richting zwart zorgt voor beter canny
    img_edges_2 = getImageEdges(img_b_gray, dilation=4, lower=120, upper=150)
    displayImage("img_b_gray", img_b_gray)
    displayImage("img_edges_1", img_edges_1)
    displayImage("img_edges_2", img_edges_2, wait=True)

    # fill in holes
    img_filled = fillInHoles(img_edges_1) # om dit juist te doen moet de edges van de bonen geen openingen hebben
    displayImage("img_filled", img_filled, wait=True)
    cv2.destroyAllWindows()

    # mark beans
    img_marked_1 = markBeanContoursOnImage(img_filled, image, find_color_alg="centroid", show_result=True)

    # mark beans
    img_marked_2 = markBeanContoursOnImage(img_filled, image, find_color_alg="contour_average", show_result=True)

    # mark beans
    img_marked_3 = markBeanContoursOnImage(img_filled, image, find_color_alg="using_mask", show_result=True)

    displayImage("result w centroid color", img_marked_1)
    displayImage("result w contour average average", img_marked_2)
    displayImage("result w using hsv mask", img_marked_3, wait=True)

    return img_marked_3

#  main
if (__name__) == "__main__":
    print("running from ws:" + str(current_dir))

    # default vals
    camera = None
    converter = None
    delay = 1
    # test_image = np.zeros((camera_w,camera_h,3), np.uint8)
    test_image = cv2.imread((images_dir+'/image1.bmp'), -1)

    # set mode
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
            test_image = cv2.imread((images_dir+'/'+input()+''), -1)

    # load settings
    json_data = None
    with open(current_dir+"/beans.json") as file:
        features = json.load(file)

    # single image
    if not (mode[0]):
        delay = 0   # wait for input

    # capture and run requires camera
    if(mode[0]):
        camera, converter = initCamera()
        capture_thread = threading.Thread(target=grabImages, args=(camera,converter))
        capture_thread.start()
        delay = 1 # 1 ms
        print("camera is running..")

    # pick a color from a hsv image and create mask
    if(mode[1]):
        # setup image
        height, width = test_image.shape[0:2]
        color_picker_image = cv2.cvtColor(test_image.copy(),cv2.COLOR_BGR2HSV)
        color_picker_image = cv2.resize(color_picker_image, ((int)(width/3)+1, (int)(height/3)+1))

        # create mask and display
        mask = np.zeros((height,width,1), np.uint8)
        displayImage("Mask Result", mask)
        displayImage("HSV_color_picker", color_picker_image, down_scale=1)

        # callback on click
        cv2.setMouseCallback("HSV_color_picker", displayMaskFromChosenPixel)

        # info
        print("click on a pixel on window 'HSV_color_picker' to create a hsv image mask")
        print("press <spacebar> on to add ranges as a feature")

    # specific modes
    if not (mode[1]):
        image_queue.put(test_image)

    if(mode[0] and not mode[1]):
        print("press <spacebar> to take image")

    if(mode[0] and mode[1]):
        delay = 1000 # 1 s

    # for util
    tick_start = 0
    tick_end = 0
    snapshot_counter = 0

    # process images
    while(True):
        image = test_image.copy()
        if not (image_queue.empty()):
            # get image
            image = image_queue.get(block=True, timeout=None)

            # timer
            tick_start = cv2.getTickCount()

            # process images
            if not (mode[1]):
                result = solution1(image)
                # result = solution2(image)
                # result = step_by_step_demo(image)
                displayImage("result", result)

            if(mode[1]):
                # create hsv image
                image_hsv = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2HSV)
                height, width = image_hsv.shape[0:2]
                color_picker_image = cv2.resize(image_hsv, ((int)(width/3)+1, (int)(height/3)+1))
                cv2.imshow("HSV_color_picker", color_picker_image)

            last_image = image.copy()
            tick_end = cv2.getTickCount()
            time_between_images = (tick_end - tick_start)/ cv2.getTickFrequency()

        # original image
        displayImage("original", image)

        # delay get get input
        key = cv2.waitKey((int)(delay))

        # spacebar
        if (key == 32):
            # add bean
            if(mode[1]):
                # get input
                print("Name of to identify feature:")
                in_str = str(input())

                # text
                new_bean = [in_str, clicked_bean_low, clicked_bean_high]
                features.append(new_bean)

                # write data
                text = []
                text.append("[")
                for i, bean_type in enumerate(features):
                    text.append("\n"+str(bean_type))
                    if(i + 1 != (len(features))):
                        text[i + 1] += ","
                text.append("]")

                # write file
                with open(current_dir+"/beans.json", "w") as file:
                    for txt in text:
                        txt = re.sub("'", '"',txt)
                        file.write(txt)

                print("saved features at "+current_dir+"/beans.json")

            # make snapshot
            else:
                snapshot_counter +=1
                path = images_dir+"/image"+ str(snapshot_counter) + ".bmp"
                cv2.imwrite(path,image)
                print("image made and saved in: " + path)

        # exit
        if (key == ord('\x1b')):
            break

    cv2.destroyAllWindows()
    if(camera):
        camera.StopGrabbing()
    sys.exit()
