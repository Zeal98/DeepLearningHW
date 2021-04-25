import cv2
import numpy as np
import json

import torch
filename = ['0lh_UWF9ZP4_27_31.avi', 'aM-RcQj0a7I_37_55.avi', 'UXs3eq68ZjE_250_255.avi']

filepath = ['./MLDS_hw2_1_data/testing_data/video/', './MLDS_hw2_1_data/training_data/video/']


def visualize_video(filepath, filename):

    vc = cv2.VideoCapture(filepath + filename)
    c = 1
    if vc.isOpened():
        print("opened")
        rval, frame = vc.read()
        print(len(frame))
        print(len(frame[0]))
        print(len(frame[0][0]))
    else:
        rval = False
    timeF = 10
    while rval:
        rval, frame = vc.read()
        if(c%timeF == 0):
            cv2.imwrite('image/'+str(c)+'.jpg', frame)
        c = c+1
    print(c)
    cv2.waitKey(1)
    vc.release()


# samples a video into a 30x256x256 matrix
def video_2_matrix(filepath, filename):
    vc = cv2.VideoCapture(filepath + filename)
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
        w = len(frame)
        h = len(frame[0])
        print(w, h, rval)
    else:
        rval = False
    timeF = 10
    frames = list()
    while rval:
        rval, frame = vc.read()
        if c % timeF == 0:
            frames.append(frame)
        c = c + 1
        if c > 1000:
            break
    # print(c)

    newFrames = list()
    for i in range(10):
        try:
            tempFrame = frames[int(len(frames)/10)]
        except:
            print(filepath, filename, len(frames))
            return False
        # print(len(frames), int(len(frames)/10))

        newFrame = [list(), list(), list()]
        for x in range(256):
            newFrame[0].append(list())
            newFrame[1].append(list())
            newFrame[2].append(list())
            for y in range(256):
                idx = int(x/256 * w)
                idy = int(y/256 * h)
                newFrame[0][x].append(int(tempFrame[idx][idy][0]))
                newFrame[1][x].append(int(tempFrame[idx][idy][1]))
                newFrame[2][x].append(int(tempFrame[idx][idy][2]))
        newFrames.append(newFrame)
    cv2.waitKey(1)
    vc.release()
    return newFrames


def prepare_training_sets(videopath, idpath):
    fp = open(idpath, 'r')
    videos = dict()
    c = 0
    for line in fp:
        c = c + 1
        if c % 100 == 0:
            print(c, 'out of 1450 is done')
            json.dump(videos, open('%straining.json'%c, 'w'))
            videos = dict()
        if line.endswith('\n'):
            templine = line[:len(line)-1]
        else:
            templine = line
        videos[templine] = video_2_matrix(videopath, templine)
    fp.close()
    json.dump(videos, open('training.json', 'w'))
    return 'training.json'
def continue_prepare(videopath, idpath, _c):
    fp = open(idpath, 'r')
    # videos = json.load(open('%straining.json'%_c, 'r'))
    videos = dict()
    c = 0
    for line in fp:
        c = c + 1
        if c <= _c:
            continue
        if c % 100 == 0:
            print(c, 'out of 1450 is done')
            json.dump(videos, open('%straining.json'%c, 'w'))
            videos = dict()
        if line.endswith('\n'):
            templine = line[:len(line) - 1]
        else:
            templine = line
        videos[templine] = video_2_matrix(videopath, templine)
    fp.close()
    json.dump(videos, open('training.json', 'w'))
    return 'training.json'


def add_dim0(c):

    videos = json.load(open('%straining.json'%c, 'r'))
    newVideos = dict()
    for video in videos.keys():
        if len(videos[video]) == 30:
            newVideos[video] = list()
            for i in range(10):
                temp = list()
                temp.append(videos[video][i * 3])
                temp.append(videos[video][i * 3 + 1])
                temp.append(videos[video][i * 3 + 2])
                newVideos[video].append(temp)
    json.dump(newVideos, open('%straining_.json'%c, 'w'))
    print(c, 'done')


if __name__ == '__main__':

    frames = video_2_matrix(filepath[0], filename[2])
    print(len(frames), len(frames[0]), len(frames[0][0]))
    for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1450]:
        add_dim0(i)

#    print(frames)
#    json.dump(frames, open('_test.json', 'w'))
#    prepare_training_sets(filepath[1], './MLDS_hw2_1_data/training_data/id.txt')
#    continue_prepare(filepath[1], './MLDS_hw2_1_data/training_data/id.txt', 100)
#    print(torch.randn(3, 4, 5))
