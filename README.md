# **Machine Learning: Fast Face & Eye Detection with Viola-Jones**

## **Quick Setup** 


**Please note this project does not support Python2. It was developed using Python 3.8.3.** <br/> **For training datasets, please refer to ``report.pdf``**. 
<br/><br/>Install the latest version of `` opencv-python `` using your package manager (e.g. ``pip``)

    $ pip install opencv-python


Install ``pandas`` and ``numpy``

    $ pip install pandas
    $ pip install numpy


To use a custom trained classifier, run ``task1.py`` from inside ``/code``

    $ python task1.py


To use the default OpenCV pre-trained classifier, run ``task2.py`` from inside ``/code``. This returns a prompt to detect faces from the ``/faces/demo`` directory, or use the webcam stream.

    $ python task2.py


To generate performance statistics, run ``eval.py`` from inside ``/code``. 

    $ python eval.py

This will compare the perfomance of all custom and OpenCV pre-trained classifiers, and output the results in a pandas dataframe. Note that this script uses a **positive** validation set.


<br/>

From _Machine Learning, Expert Systems and Fuzzy Logic_ (ICS3206) of the [B.Sc. Computer Science course.](https://www.um.edu.mt/courses/overview/UBSCHICGCFT-2020-1-O)
