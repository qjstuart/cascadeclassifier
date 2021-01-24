import cv2 as cv
import os

def main():

    # select cascade classifier and start webcam stream
    cascade_face = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0995mFA045stages14/cascade.xml')
    cam = cv.VideoCapture(0)

    while(True):
        ret, frame = cam.read() 
        frame = cv.flip(frame, 1)

        # object detection, returns list of rectangles
        rectangles = cascade_face.detectMultiScale(frame)

        for(x, y, w, h) in rectangles:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 100), 3)
  
        # display the resulting frame 
        cv.imshow('frame', frame) 
        
        # press q to quit
        if cv.waitKey(1) & 0xFF == ord('q'): 
            break
    
    cam.release() 
    # Destroy all the windows 
    cv.destroyAllWindows()


def make_neg():
    with open('neg.txt', 'w') as file:
        for filename in os.listdir('negative'):
            file.write('negative/' + filename + '\n')


def make_pos():
    with open('pos.txt', 'w') as file:
        for filename in os.listdir('positive'):
            file.write('positive/' + filename + ' 1 0 0 64 64' + '\n')


if __name__ == "__main__":
    main()
