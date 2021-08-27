import sys
#import cv2
import datetime
import time

from PIL import Image
import numpy as np

from decode import VideoDecord
from mxnet import gluon, nd, image
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms import video
from queue import Queue



def STARTVideo(filename):

    #cap = cv2.VideoCapture(filename)

    vr = VideoDecord(filename)
    ci = vr.detect()
    model_name = 'slowfast_4x16_resnet50_kinetics400'
    net = get_model(model_name, nclass=400, pretrained=True)
    pred = net(nd.array(ci))
    allval = []
    classes = net.classes
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    for i in range(topK):
        val = (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar())
        allval.append(val)
    # print(allval[0][0])

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # date = datetime.datetime.now()
    # out = cv2.VideoWriter(f'data/Video_{date}.avi', fourcc, 30, (int(cap.get(3)),int(cap.get(4))
    #                                                                 ))

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = 1
    # # Blue color in BGR
    # color = (255, 0, 0)

    # # Line thickness of 2 px
    # thickness = 2
    # while(True):
    #     ret, frame = cap.read()
    #     if ret == True:
    #         cv2.putText(frame, allval[0][0], (20, 40) , font, fontScale,color, thickness, cv2.LINE_AA, False)
    #         out.write(frame)


    #         if(cv2.waitKey(25) == ord('q')):
    #             break
    #     else:
    #         break
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

    return allval[0][0]
