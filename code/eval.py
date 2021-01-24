import cv2 as cv
import os
import pandas as pd

eval_dir = ".\\faces\\eval"     # evaluation images directory
eval_pos =   False             # evaluation mode


# define columns 
columns = ["true positives", "true negatives", "false positives", "false negatives", "precision", "recall", "f-measure"]
# define index
index = ["opencv_frontalface_default", "opencv_frontalface_alt", "opencv_frontalface_alt2", "pos500neg1000stages11mhr0995mfar050", "pos1000neg500stages13mhr0995mfar050", "pos1500neg3000mhr0995mfar050stages13", "pos1500neg3000mhr0996mfar050stages13", "pos1500neg3000mhr0997mfar050stages13", "pos1500neg3000mhr0998mfar050stages12", "pos1500neg3000mhr0999mfar050stages13", "pos1500neg3000mhr0995mfar045stages14", "pos1500neg3000mhr0995mfar040stages12", "pos1500neg3000mhr0995mfar035stages11", "pos1500neg3000mhr0995mfar030stages10"]

results = pd.DataFrame(index=index, columns=columns)    # make pandas dataframe
results = results.fillna(0)                             # replacec empty values with 0

# load all cascades
opencv_frontalface_default = cv.CascadeClassifier('cascade/task2/haarcascade_frontalface_default.xml')
opencv_frontalface_alt = cv.CascadeClassifier('cascade/task2/haarcascade_frontalface_alt.xml')
opencv_frontalface_alt2 = cv.CascadeClassifier('cascade/task2/haarcascade_frontalface_alt2.xml')
classifier1 = cv.CascadeClassifier('cascade/task1/pos500neg1000stages11/cascade.xml')
classifier2 = cv.CascadeClassifier('cascade/task1/pos1000neg500stages13/cascade.xml')
classifier3 = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0995mFA050stages13/cascade.xml')
classifier4 = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0996mFA050stages13/cascade.xml')
classifier5 = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0997mFA050stages13/cascade.xml')
classifier6 = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0998mFA050stages12/cascade.xml')
classifier7 = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0999mFA050stages13/cascade.xml')
classifier8 = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0995mFA045stages14/cascade.xml')
classifier9 = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0995mFA040stages12/cascade.xml')
classifier10 = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0995mFA035stages11/cascade.xml')
classifier11 = cv.CascadeClassifier('cascade/task1/pos1500neg3000/mHR0995mFA030stages10/cascade.xml')


# add cascasdes to list for iteration
cascades = [opencv_frontalface_default, opencv_frontalface_alt, opencv_frontalface_alt2, classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, classifier7, classifier8, classifier9, classifier10, classifier11]

# try:
for filename in os.listdir(eval_dir):
    
    # read in 1 frame
    frame = cv.imread(eval_dir + "\\" + filename)   

    # convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # pass grayscale frame through all loaded cascades 
    for cascade_name, cascade in zip(index, cascades):
        
        # attempt face detection, returns list of rectangles
        faces = cascade.detectMultiScale(gray, 1.3, 5)      
        
        # for positives evaluation
        if eval_pos == True:

            # if 1 or more rectangles returned
            if len(faces) >= 1:
                results.at[cascade_name,"true positives"] += 1                  # add true positive
                results.at[cascade_name, "false positives"] += len(faces) - 1   # (1 + x) faces are detected, add x as false positives
            
            # if no rectangles returned (no faces detected)
            if len(faces) < 1:
                results.at[cascade_name, "false negatives"] += 1                # add a false negative

        # for negatives evaluation
        elif eval_pos == False:
            
            # if 1 or more rectangles returned
            if len(faces) >= 1:
                results.at[cascade_name, "false positives"] += len(faces)       # add false positives
            
            # if no rectangles returned (no faces detected)
            if len(faces) < 1:
                results.at[cascade_name, "true negatives"] += 1                 # add true negative

# except:
#     print("[-] Could not locate folder " + eval_dir)
#     exit()

# compute precision, recall and f-measure
results["precision"] = results["true positives"] / (results["true positives"] + results["false positives"])
results["recall"] = results["true positives"] / (results["true positives"] + results["false negatives"])
results["f-measure"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"])

# show results dataframe
print(results)