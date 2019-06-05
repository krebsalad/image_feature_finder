# author:
# Johan Mgina
# Date modified: 5-6-2019
# version: 0.8
# Todo next: relative directories and also arguments

from pypylon import pylon
import cv2
import time
import sys
import numpy as np
from matplotlib import pyplot as plt
import threading
import queue

# image queue vals
image_queue = queue.Queue()
q_size = 3
time_between_images = 0

# color picker vals
image_color_picker = None
bean_color_ranges = {
    'kidney_boon_color_low': np.array([131,  75,  58]    ),   'kidney_boon_color_high': np.array([151,  95, 138]),
    'koffie_boon_color_low': np.array([115,  37,  31] ),     'koffie_boon_color_high': np.array([135,  57, 111]),
    'bruine_boon_color_low': np.array([  9, 149, 105]),     'bruine_boon_color_high': np.array([ 29, 169, 185]),
    'zwarte_boon_color_low': np.array([ 97, 157,  58]),     'zwarte_boon_color_high': np.array([117, 177, 138]),
    }


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
                time.sleep(1)

    print("camera stopped!!")

def initCamera():
    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    #set the dimentions og the image to grab
    camera.Open()
    camera.Width.Value = 2074  # 0.8% max width of Basler puA2500-14uc camera
    camera.Height.Value = 1554 # 0.8% max height of Basler puA2500-14uc camera
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

def displayMaskFromChosenColor(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_color_picker[y,x]
        print("picked color:", pixel)

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print("color ranges:", lower, upper)

        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS
        image_mask = cv2.inRange(image_color_picker,lower,upper)
        displayImage("Mask result", image_mask, down_scale=1)


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
    img_hsv = cv2.GaussianBlur(img_original , (27,27), 3)
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)

    # create mask
    image_mask_kidney = None
    image_mask_zwarte = None
    image_mask_bruine = None
    image_mask_koffie = None
    if(find_color_alg == "using_mask"):
        image_mask_kidney = cv2.inRange(img_hsv, bean_color_ranges["kidney_boon_color_low"],bean_color_ranges["kidney_boon_color_high"])
        image_mask_zwarte = cv2.inRange(img_hsv, bean_color_ranges["zwarte_boon_color_low"],bean_color_ranges["zwarte_boon_color_high"])
        image_mask_bruine = cv2.inRange(img_hsv, bean_color_ranges["bruine_boon_color_low"],bean_color_ranges["bruine_boon_color_high"])
        image_mask_koffie = cv2.inRange(img_hsv, bean_color_ranges["koffie_boon_color_low"],bean_color_ranges["koffie_boon_color_high"])

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
                if(isHsvColorInRange(pixel_color, bean_color_ranges["kidney_boon_color_low"], bean_color_ranges["kidney_boon_color_high"])):
                    img_contours = writeTextOnImage(img_contours, "kidney boon", centroid_x, centroid_y)
                    print_text += " - Type: kidney boon "

                # koffie bonen
                if(isHsvColorInRange(pixel_color, bean_color_ranges["koffie_boon_color_low"], bean_color_ranges["koffie_boon_color_high"])):
                    img_contours = writeTextOnImage(img_contours, "koffie boon", centroid_x, centroid_y)
                    print_text += " - Type: koffie boon "

                # bruine boon
                if(isHsvColorInRange(pixel_color, bean_color_ranges["bruine_boon_color_low"], bean_color_ranges["bruine_boon_color_high"])):
                    img_contours = writeTextOnImage(img_contours, "bruine boon", centroid_x, centroid_y)
                    print_text += " - Type: bruine boon "

                # zwarte boon
                if(isHsvColorInRange(pixel_color, bean_color_ranges["zwarte_boon_color_low"], bean_color_ranges["zwarte_boon_color_high"])):
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


# solutions
def solution1(image):
    # create grayscaled image
    img_gray, img_b_gray, img_g_gray, img_r_gray = getBgrGrayscaledImages(image)

    # edges
    img_edges = getImageEdges(img_b_gray, dilation=2, lower=20, upper=50)

    # fill in holes
    img_filled = fillInHoles(img_edges)

    # mark beans
    img_marked = markBeanContoursOnImage(img_filled, image, find_color_alg="using_mask") # contour_average using_mask

    return img_marked

def solution2(image):
    # create grayscaled image
    img_gray, img_b_gray, img_g_gray, img_r_gray = getBgrGrayscaledImages(image)

    # edges
    img_median = markPixelsAboveMedian(img_gray)
    displayImage("median", img_median)

    # close image
    # img_closed = closeImageEdges(img_median)
    # displayImage("closed", img_closed)

    # fillInHoles
    img_filled = fillInHoles(img_median)
    displayImage("filled", img_filled)

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
    # for timing
    tick1 = 0
    tick2 = 0
    delay_ms = 1 # delay between images

    # camera
    camera = None
    converter = None
    img_counter = 0

    # parse/read/set args
    mode = "test" # default
    test_image = cv2.imread('/home/iot/practicum_opdrachten/opgaven_iot_vis/images/Bonen_110k_diaf_8_gain_0.bmp', -1)
    num_args = len(sys.argv)

    if(num_args > 1):
        mode = sys.argv[1]

    if(num_args > 2):
        test_image = cv2.imread(sys.argv[2], -1)

    # set test image
    if(mode == "test"):
        image_queue.put(test_image)

    # capture and run requires camera
    if(mode == "run" or mode == "view" or mode == "pick_color"):
        camera, converter = initCamera()
        capture_thread = threading.Thread(target=grabImages, args=(camera,converter))
        capture_thread.start()

        # pick a color from a hsv image and create mask
        if(mode == "pick_color"):
            # pick color cb on HSV image with window named HSV
            image_color_picker = test_image.copy()
            cv2.imshow("HSV_color_picker", image_color_picker)
            cv2.setMouseCallback("HSV_color_picker", displayMaskFromChosenColor)

        if(mode == "view"):
            print("press <spacebar> to take image")


    # processing thread
    while(True):
        # find beans if image available
        if not (image_queue.empty()):

            # get image
            image = image_queue.get(block=True, timeout=None)

            # timer
            tick1 = cv2.getTickCount()

            # process images
            if(mode == "run" or mode == "test"):
                result = solution1(image)
                # result = solution2(image)
                # result = step_by_step_demo(image)
                displayImage("result", result)
                displayImage("original", image)

            if(mode == "pick_color"):
                # create hsv image
                image_hsv = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2HSV)
                down_scale = 3
                height, width = image_hsv.shape[0:2]
                image_color_picker = cv2.resize(image_hsv, ((int)(width/down_scale)+1, (int)(height/down_scale)+1))
                cv2.imshow("HSV_color_picker", image_color_picker)

            if(mode == "view"):
                # original image
                displayImage("view", image)
                k = cv2.waitKey(1)
                if (k == 32):
                    img_counter +=1
                    path = "/home/iot/practicum_opdrachten/opgaven_iot_vis/images/image" + str(img_counter) + ".bmp"
                    cv2.imwrite(path,image)
                    print("image made and saved in: " + path)

            # print time took
            tick2 = cv2.getTickCount()
            time_between_images = (tick2 - tick1)/ cv2.getTickFrequency()
            # print("Time took to prosess image:" + str(time_between_images) + "seconds")

        # delay between processing images and exit (escape)
        key = cv2.waitKey((int)(1))
        if (key == ord('\x1b')):
            break


    # exit
    cv2.destroyAllWindows()
    if(camera):
        camera.StopGrabbing()
    sys.exit()
