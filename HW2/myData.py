import torch
import json
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

class myData(Dataset):
    def __init__(self, videofile, labelfile):
        videoSet = json.load(open(videofile, 'r'))
        labelSet = json.load(open(labelfile, 'r'))
        self.videos = list()
        self.x_data = list()
        self.y_data = list()

        i = 0
        for video in labelSet:

            video_id = video['id']
            caps = video['caption']

            try:
                frames = videoSet[video_id] #10, 3, 256, 256
            except:
                continue

            self.videos.append(torch.Tensor(frames))
            i = len(self.videos) - 1
            for cap in caps:
                self.x_data.append(i)   #index to the video
                self.y_data.append(torch.Tensor(cap))

    def __getitem__(self, index):
        return self.videos[self.x_data[index]], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

if __name__ == '__main__':
    c = 100
    dataset = myData('%straining.json'%c, './MLDS_hw2_1_data/training_vec.json')
