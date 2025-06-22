mobile.py is to detect any mobile phone presence in frame or not
face.py is to check if ssame erson is present or not
eye.py is for eye grazing movement left right up down centre
headpose.py is for head tracking left right up down centre

mobile.py is not created on my system. it is created and tested in google colab so maybe some dependencies issues maybe there.
same with face.py

in eye.py and head.py the model expects the mirror/reflected image taken from webcam of my laptop on the based of which it tells the correct position of person head. for eg if head is in left, the webacm will save 
mirror image of head in right but model will output left only.

please use python version 3.10.3 as mediappipe is not supported in new versions.
