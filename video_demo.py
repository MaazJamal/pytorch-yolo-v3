from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse



def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def save_frame_data(x,classNumber):

    cls = int(x[-1])
    if cls in classNumber:
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        label = classes[cls]
        text_line = ("{0},{1},{2},{3},{4}".format(label, c1[0], c1[1], c2[0], c2[1]))
        return text_line


def class_numbers(class_number):

    try:
        f = open(class_number, 'r')
    except IOError:
        print("could not open class file:")
        sys.exit(1)
    with f:
        temp = f.readline().split(',')
        temp[-1] = temp[-1].strip('\n')
        return tuple(map(int, temp))



def arg_parse():
    """
    Parse arguments to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest='video', help=
                        "Video to run detection upon",
                        default="./video", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
                        "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
                        "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="64", type=str)
    parser.add_argument("--class", dest='class_number', help=
                        "input .txt file with class number from coco dataset that you want to output",
                        default="data/classNumbers.txt", type=str)
    parser.add_argument("--buffer", dest='bufferSize', help=
                        "Size of the buffer to store the file to memory", default=2, type=int)
    parser.add_argument("--savepath", dest='savePath', help=
                        'The place where you want to save the dataset',
                        default='./dataset', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    classNumbers = class_numbers(args.class_number)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()
    
    classes = load_classes('data/coco.names')

    video_path = args.video
    videofiles = video_file_parser(video_path)
    count = 0
    print("these are video files: {0}".format(videofiles))
    for videofile in videofiles:
        print(videofile)
        cap = cv2.VideoCapture(videofile)

        assert cap.isOpened(), 'Cannot capture source'
        buff = args.bufferSize
        path = args.savePath
        output_handler = FileHandle(path, buff)
        frames = 0
        start = time.time()
        print("this is count {0}".format(count))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:

                img, orig_im, dim = prep_image(frame, inp_dim)

                im_dim = torch.FloatTensor(dim).repeat(1, 2)

                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()

                with torch.no_grad():
                    output = model(Variable(img), CUDA)
                output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

                if type(output) == int:
                    frames += 1
                    print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                   # print("It in the if statement, But why though? This is output: {0}".format(output))
                   # cv2.imshow("frame", orig_im)
                   # key = cv2.waitKey(1)
                   # if key & 0xFF == ord('q'):
                   #     break
                    continue

                im_dim = im_dim.repeat(output.size(0), 1)
                scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)

                output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
                output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

                output[:, 1:5] /= scaling_factor

                for i in range(output.shape[0]):
                    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

                colors = pkl.load(open("pallete", "rb"))

            #    print(classNumbers)
                textLine = list(map(lambda x: save_frame_data(x, classNumbers), output))
            #    print(textLine)
                output_handler.add_frame(orig_im, textLine)
            #    list(map(lambda x: write(x, orig_im), output))
            #    print(output)
            #    cv2.imshow("frame", orig_im)
            #    key = cv2.waitKey(1)
            #    if key & 0xFF == ord('q'):
            #        break

                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))

            else:
                break
            print("This is frames: {0}".format(frames))
            if frames % 4 == 0:
                count += 1
                break
        cap.release()