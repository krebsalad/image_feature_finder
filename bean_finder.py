# example finder of feature_finder

from pypylon import pylon
import cv2
import time
import sys
from threading import Thread
import os
import glob
from sequential_item_buffer import *
from feature_finder import *

# global
global exit_threads
exit_threads = False

# camera
def initCamera():

    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera_w = 1554
    camera_h = 2074

    #set the dimentions og the image to grab
    camera.Open()
    camera.Width.Value = camera_h  # 0.8% max width of Basler puA2500-14uc camera
    camera.Height.Value =  camera_w# 0.8% max height of Basler puA2500-14uc camera
    camera.OffsetX.Value = 518
    # camera.AcquisitionFrameRate.SetValue(14)

    # set features of camera
    camera.ExposureTime.Value = 110000
    camera.ExposureAuto.SetValue('Off')
    camera.BalanceWhiteAuto.SetValue('Off')
    camera.LightSourcePreset.SetValue('Daylight5000K')
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

def grabImagesThread(camera, converter, image_buffer):
    global exit_threads
    img_counter = 0
    while camera.IsGrabbing() and not exit_threads:

        # check if list is not full
        is_len, length = image_buffer.getLength(timeout=1000)
        if(is_len):
            if(length < 2):

                # grab image
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    image = converter.Convert(grabResult)
                    img = image.GetArray()
                    image_buffer.putLast((img_counter, img))
                    img_counter += 1
                grabResult.Release()

    print("camera stopped!!")

# display images
def displayBuffer(buffers_in, start_sq_nr=0, frame_time_sec=1, exit_key=ord('\x1b')):
    # show next image if time passed
    displayed_images_counter = start_sq_nr
    len_buffers = len(buffers_in)
    tick_start = cv2.getTickCount()
    to_display_images = [None]*len_buffers
    task_done = [False]*len_buffers
    while not exit_threads:
        done = True
        for i, buffer_in in enumerate(buffers_in):
            if not task_done[i]:
                is_img, image = buffer_in.getImageUsingSeqNr(displayed_images_counter, timeout=33)
                if is_img:
                    # resize
                    h, w = image[1].shape[0:2]
                    resized_img = cv2.resize(image[1], ((int)(w/3)+1,(int)(h/3)+1))
                    to_display_images[i] = resized_img
                    task_done[i] = True
                else:
                    done = False

            # exit thread
            if(cv2.waitKey(1) == exit_key):
                return

        if(done):
            # timer, sleep for time left between frames
            tick_end = cv2.getTickCount()
            time_took = (tick_end - tick_start)/ cv2.getTickFrequency()
            sleep_time = (frame_time_sec - time_took)
            if(sleep_time > 0):
                time.sleep(sleep_time)
            tick_start = cv2.getTickCount()

            # if all images are gotten
            for i, buffer_in in enumerate(buffers_in):
                # show
                cv2.imshow(buffer_in.name+"_window", to_display_images[i])
                print("display_"+buffer_in.name+": displayed image"+str(displayed_images_counter))

                # exit thread
                if(cv2.waitKey(1) == exit_key):
                    return

            # next
            displayed_images_counter +=1
            to_display_images = [None]*len_buffers
            task_done = [False]*len_buffers

        # exit thread
        if(cv2.waitKey(1) == exit_key):
            return

# print stuff
def printHelp():
    print("running from ws:" + str(os.getcwd()))
    help_str = "This is a tool to find ellipse like objects with color on an image with a white background\n"
    help_str += "Options:\n\n"
    help_str += "with_image : (prompt) will ask for image in /images/\n"
    help_str += "with_camera : run contiounsly with camera\n"
    help_str += "with_images : read directory: /images/\n\n"
    help_str += "add_feature : add feature from hsv range to /features.json (loads on start)\n"
    help_str += "find_features: find features on given image using features defined in file /features.json\n\n"
    help_str += "verbose : print and display more images in between processing, for debugging\n\n"
    help_str += "result_to_queue: output results into combined buffer, will use a seprate thread to display images\n"
    print(help_str)

#  main
if (__name__) == "__main__":
    ###################
    ## image buffers ##
    image_buffer = SequentialItemBuffer("raw_images")
    result_image_buffer = SequentialItemBuffer("result_images")
    original_image_buffer =  SequentialItemBuffer("original_images")
    sec_per_frame = 1    # for display 1 fps
    ####################


    ###########################
    ## Setup finder settings ##
    mode = [False, False, False, False]
    verbose = False
    # mode[0] = single_view(False) or continous_view(True)
    # mode[1] = view_image(False) or process image(True)
    # mode[2]  = view only(False) or add_feature(True)
    # mode[3] = use_display(False) or use_queues/no_display(True)
    # verbose = use display for debugging and extra info, only if display allowed

    # print help
    if(len(sys.argv) == 1):
        printHelp()
        sys.exit()

    # read args and images
    img_counter = 0
    for arg in sys.argv:
        if((arg == "h" or arg == "help" or arg == "-h") and len(sys.argv) == 2):
            printHelp()
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
                image_buffer.putLast((img_counter, cv2.imread(img_pth, -1)))
                img_counter+=1
            else:
                print("could not find image"+img_pth)
                print("exiting")
                sys.exit()

        if(arg == "with_images"):
            img_pth = (os.getcwd()+'/images/')
            if(os.path.isdir(img_pth)):
                image_paths = glob.glob(img_pth+"*.bmp")
                if(len(image_paths) > 0):
                    for pth in image_paths:
                        image_buffer.putLast((img_counter,cv2.imread(pth, -1)))
                        img_counter+=1
                        print("loaded image:"+pth)
                else:
                    print("no image was found in "+ img_pth)
                    print("exiting")
                    sys.exit()
            else:
                print("directory "+ img_pth +" does not exist!")
                print("exiting")
                sys.exit()

        if(arg == "result_to_queue"):
            mode[3] = True # multi threading (will disable view from feature finder == no add_Feature and no display and always continous)

        if(arg == "verbose"):
            verbose = True
    ############################


    ###################
    ## MAIN PROGRAM ##
    # start camera if needed
    capture_thread = None
    camera = None
    converter = None
    if(mode[0]):
        sec_per_frame = 0.20
        camera, converter = initCamera()
        capture_thread = Thread(target=grabImagesThread, args=(camera,converter,image_buffer))
        capture_thread.start()

    # setup finder and start finder
    bean_finder = FeatureFinder(mode, image_buffer, result_image_buffer=result_image_buffer, original_image_buffer=original_image_buffer, name="bean_finder_1", verbose=verbose)
    finder_thread = Thread(target=bean_finder.run)
    finder_thread.start()
    ################


    ############################
    ## Extra features mode[3] ##
    # using multiple threads
    if(mode[3]):
        bean_finder_2 = None
        finder_thread_2 = None
        bean_finder_3 = None
        finder_thread_3 = None

        # setup theads for displaying in case of extra finders
        display_thread = None
        display_thread_2 = None

        # extra processing thread
        bean_finder_2 = FeatureFinder([True, True, False, True], image_buffer, result_image_buffer=result_image_buffer, original_image_buffer=original_image_buffer, name="bean_finder_2")
        finder_thread_2 = Thread(target=bean_finder_2.run)
        finder_thread_2.start()
        bean_finder_3 = FeatureFinder([True, True, False, True], image_buffer, result_image_buffer=result_image_buffer, original_image_buffer=original_image_buffer, name="bean_finder_3")
        finder_thread_3 = Thread(target=bean_finder_3.run)
        finder_thread_3.start()

        # display
        buffers = [result_image_buffer, original_image_buffer]
        displayBuffer(buffers,start_sq_nr=0, frame_time_sec=sec_per_frame, exit_key=ord('\x1b'))

        # join the threads
        bean_finder.stop = True
        bean_finder_2.stop = True
        bean_finder_3.stop = True
        print("trying to exit threads")

        finder_thread_2.join()
        finder_thread_3.join()
    ############################


    ##########
    ## Exit ##
    finder_thread.join()
    exit_threads = True
    if(mode[0]):
        capture_thread.join()
        if(camera):
            camera.StopGrabbing()
    print("exited succesfully")
    cv2.destroyAllWindows()
    sys.exit()
