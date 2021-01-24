import cv2 as cv
import numpy as np
import os
import time, subprocess

def main():

    # load the OpenCV face and eye classifiers 
    face_cascade = cv.CascadeClassifier('cascade/task2/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('cascade/task2/haarcascade_eye.xml')

    # display menu
    welcome()
 
    while(True):
        # prompt user for mode
        prompt = input("facedetector> ")

        if (prompt == "d"):
            detect_from_folder(face_cascade, eye_cascade)
        elif (prompt == "w"):
            print("facedetector> [+] Staring webcam stream...")
            try:
                webcam_stream(face_cascade, eye_cascade)
            except:
                print("facedetector> [-] Could not access webcam, maybe it is in use?")
        elif (prompt == "q"):
            print("facedetector> [+] Quitting...")
            time.sleep(1)
            try: 
                subprocess.check_output('cls', shell=True)
                os.system('cls')
            except subprocess.CalledProcessError as e:
                try: 
                    subprocess.check_output('clear', shell=True)
                    os.system('clear')
                except subprocess.CalledProcessError as e:
                    break

            break
        else:
            print("facedetector> [-] Not an option")



def detect_from_folder(f, e):

    images = []         
    folder = ".\\faces\\demo"
    
    try:
        # load demo images into images list
        for filename in os.listdir(folder):
            i = cv.imread(folder + "\\" + filename)
            if i is not None:
                images.append(i)
    except:
        print("facedetector> [-] Could not locate folder " + folder)
        return

    # images found
    print("facedetector> [+] Detecting faces from " + folder + "...")

    # classify loaded images
    for img in images:
        
        # convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # detect faces using recommended values for scale factor and minimum neighbors (1.3 & 1.5)
        faces = f.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            img = cv.rectangle(img,(x,y), (x+w,y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # detect eyes within face region
            eyes = e.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0, 0, 100), 3)


        cv.imshow('image', img)

        # press 'q' to quit
        if cv.waitKey(0) == ord('q'): 
            cv.destroyAllWindows()
            break

        else:
            # display results onscreen
            cv.destroyAllWindows()

        

    
def webcam_stream(f, e):

    # set up webcam stream
    cam = cv.VideoCapture(0)

    # continuously get webcam frames until user hits 'q'
    while(True):
        ret, frame = cam.read()     # get frame
        frame = cv.flip(frame, 1)   # filp frame horizontally
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert frame to grayscale

        # detect faces
        faces = f.detectMultiScale(gray_frame)

        for(x, y, w, h) in faces:
            frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # draw red rectangle around each detected face
            cv.putText(frame, 'face', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            roi_gray = gray_frame [y:y+h, x:x+w]
            roi_color = frame [y:y+h, x:x+w]

            # detect eyes within face region
            eyes = e.detectMultiScale(roi_gray)                   
            for (ex,ey,ew,eh) in eyes:
                eye_rect = cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (107, 107, 255), 3)  # draw rectangle around each detected eye
                cv.putText(eye_rect, 'eye', (ex, ey-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (107, 107, 255), 3)

             
        # display the resulting frame 
        cv.imshow('frame', frame) 
        
        # press 'q' to quit
        if cv.waitKey(1) & 0xFF == ord('q'): 
            break
    
    cam.release() 
    cv.destroyAllWindows()      # destroy all windows 



def welcome():

    ret = print(""" 
     #####
    #### _\_  ________
    ##=-[.].]| \      \\
    #(    _\ |  |------|
     #   __| |  ||||||||
      \  _/  |  ||||||||
   .--'--'-. |  | ____ |
  / __      `|__|[o__o]|
_(____nm_______ /____\____

----------------------------------------------------------------
Welcome to the Face Detector, which hopefully, detects your face
----------------------------------------------------------------
Where shall I look for faces?   [d] faces directory
-----------------------------   [w] webcam stream
                                [q] quit""")

# This ASCII pic can be found at
# https://asciiart.website/index.php?art=people/faces

    return ret



if __name__ == "__main__":
    main()
