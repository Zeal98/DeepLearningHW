import torch
import torch.nn.functional as F
import json
import videoLoader as vL
import dictBuilder as dB
import myData

use_gpu = torch.cuda.is_available()
# global embed_
embed_ = torch.nn.Embedding(3000, 256)
embed_.load_state_dict(torch.load('./models/embed_.pth'))

class MyLSTM_1(torch.nn.Module):
    def __init__(self):
        # global embed_
        super(MyLSTM_1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 5, padding=2)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(12, 48, 3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(4)
        self.line1 = torch.nn.Linear(49152, 4096)

        self.lstm1 = torch.nn.LSTM(4096, 256)
        # self.lstm2 = torch.nn.LSTM(49152, 256)
        self.lstmc1 = torch.nn.LSTMCell(256, 256)
        self.lstmc2 = torch.nn.LSTMCell(256, 256)
        self.hidden_cell1 = (torch.zeros(1, 1, 256), torch.zeros((1, 1, 256)))
        self.hidden_cell2 = (torch.zeros(1, 1, 256), torch.zeros((1, 1, 256)))
        # self.line2 = torch.nn.Linear(1024, 256)

        # self.embed = torch.nn.Embedding(3000, 256)

    def forward(self, video):
        # global embed_
        if use_gpu:
            embed = embed_.cuda()
        else:
            embed = embed_
        #each video contains 10 cap frames, thus batch = 10
        out = F.relu(self.conv1(video)) # size() = (10, 3, 256, 256)
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size()[0], -1)
        out = F.relu(self.line1(out))

        # t = out.transpose(0, 1)
        out = out.unsqueeze(1)  # size() = (10, 1, 512)
        # print(out.size())

        #"""
        if use_gpu:
            h, c0 = self.hidden_cell1[0].cuda(), self.hidden_cell1[1].cuda()
        else:
            h, c0 = self.hidden_cell1[0], self.hidden_cell1[1]
        out, (h, c0) = self.lstm1(out, (h, c0))
        # out, (h, c) = self.lstm2(out, h)

        # print(out.size(), h.size())
        # 10, 1, 256
        temp = torch.LongTensor([3000-1])

        coll = torch.zeros(18, 256)
        c1 = torch.zeros(1, 256)
        if use_gpu:
            temp, coll, c1 = temp.cuda(), coll.cuda(), c1.cuda()
        temp = embed(temp)
        h = h[0]
        for i in range(10):
            # print(out[i].size(), temp.size(), c1.size())
            temp, c1 = self.lstmc1(out[i], (temp, c1))
        h = temp
        temp = torch.zeros(1).long()
        if use_gpu:
            temp = temp.cuda()
        temp = embed(temp)    #<BOS>
        for i in range(18):
            temp, c1 = self.lstmc2(temp, (h, c1))
            coll[i] = temp

        """ 
        h, c0 = self.hidden_cell1[0], self.hidden_cell1[1]
        h, c0, c1 = (h[0], c0[0], c0[0])
        h1 = torch.zeros(1, 256)
        coll = torch.zeros(18, 256)
        padding = torch.LongTensor([3000-1])
        if use_gpu:
            h, c0, h1, c1, coll, padding = h.cuda(), c0.cuda(), h1.cuda(), c1.cuda(), coll.cuda(), padding.cuda()
        padding = embed(padding)
        for i in range(5):
            # print(out.size(), h.size(), c0.size(), padding.size(), h.size(), c1.size())
            h, c0 = self.lstmc2(out[i], (h, c0))
            h1, c1 = self.lstmc1(padding, (h, c1))
        h1 = torch.LongTensor([0])
        if use_gpu:
            h1 = h1.cuda()
        h1 = embed(h1)

        for i in range(18):
            h, c0 = self.lstmc2(out[i%10], (h, c0)) # how to do padding for frame input?
            h1, c1 = self.lstmc1(h1, (h, c1))
            coll[i] = h1
        """
        # ret = torch.zeros(18)
        # for i in range(18):
        #     dist = torch.norm(self.embed.weight.data - coll[i], dim=1)
        #     ret[i] = torch.argmin(dist)
        # return ret

        return coll  # (18, 256)


def train(num_epoch, videofile, labelfile, savepath='./models/all_states.pth'):
    # global embed_
    if use_gpu:
        embed = embed_.cuda()
    else:
        embed = embed_
    # frozen_list = ['embed']
    dataset = myData.myData(videofile, labelfile)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # dictfile = json.load(open(dictfile), 'r')
    model = MyLSTM_1()
    try:
        saved = torch.load(savepath)
        model.load_state_dict(saved["net"])

    except:
        saved = False
        print("No existing model found")
        print("Starting a new model")

    # for name, value in model.named_parameters():
    #     if name in frozen_list:
    #         value.requires_grad = False
    # params = filter(lambda p: p.requires_grad, model.parameters())
    loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if use_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    for t in range(num_epoch):
        print("Current epoch:", t)
        count = 0
        for i, data in enumerate(dataloader, 0):
            count += 1
            inputs, labels = data
            # if count%10 == 0:
               # print(inputs.size(), labels)
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs[0])
            labels = labels.long()

            labels = embed(labels[0].long())
            # labels = labels[0]
            #print(outputs.size(), labels.size())
            loss = loss_fn(outputs, labels)
            if count % 50 == 0:
                print("count = %s, loss = %s"%(count,loss))
            loss.backward()
            optimizer.step()

        if t % 5 == 0:
            all_states = {"net": model.state_dict(), "opt": optimizer, "epoch": t}
            torch.save(obj=all_states, f=savepath)

    all_states = {"net": model.state_dict(), "opt": optimizer, "epoch": t}
    torch.save(obj=all_states, f=savepath)
    return all_states


def testing(testsetpath, dictpath, modelpath, outputpath):
    global use_gpu
    use_gpu = False
    embed = embed_
    # "./MLDS_hw2_1_data/testing_data/", "temp.json", "./models/0.pth", "./MLDS_hw2_1_data/pred.txt"
    videos = dict()
    fp = open(testsetpath + "id.txt", 'r')
    videopath = testsetpath + "video/"
    for line in fp:
        if line.endswith('\n'):
            templine = line[:len(line) - 1]
        else:
            templine = line
        videos[templine] = vL.video_2_matrix(videopath, templine)
    fp.close()

    model = MyLSTM_1()
    try:
        saved = torch.load(modelpath)
        model.load_state_dict(saved["net"])
        for name, paras in model.named_parameters():
            print(name, paras)
    except:
        print("model not found!")
        return False

    preds = dict()
    words = json.load(open(dictpath, 'r'))
    words_re = dict()
    words_re[3000 - 1] = '<Unknown>'
    words_re[3000 - 2] = '<Pad>'
    words_re[0] = '<BOS>'
    words_re[1] = '<EOS>'
    for word in words.keys():
        words_re[words[word]] = word

    out = dict()
    for video in videos.keys():
        preds[video] = model(torch.Tensor(videos[video]))
        one_hots = torch.zeros(18)
        sent = list()
        for i in range(18):
            dist = torch.norm(embed.weight.data - preds[video][i], dim=1)
            one_hots[i] = torch.argmin(dist)
            if int(one_hots[i]) in words_re.keys():
                sent.append(words_re[int(one_hots[i])])
            else:
                sent.append("<Unknown>")
        out[video] = ""
        for word in sent:
            if (word == "<Unknown>") or word == "<Pad>":
                continue
            if word == "<EOS>":
                break
            out[video] = out[video] + ' ' + word

    fp = open(outputpath, 'w')
    for video in out.keys():
        print(video, out[video], file=fp)
    fp.close()
    use_gpu = torch.cuda.is_available()
    return out


if __name__ == "__main__":
    # c = 1450
    comm = "train"
    if comm == "train":
        cl = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1450)
        for i in range(10):
            for c in cl:
                print("Current c =", c)
                videofile = "%straining.json"%c
                labelfile = "./MLDS_hw2_1_data/training_vec.json"
                states = train(2, videofile, labelfile, savepath='./models/all_states_4.pth')
            torch.save(obj=states, f='./models/4_%s.pth'%i)
    if comm == "test":
        testing("./MLDS_hw2_1_data/testing_data/", "temp.json", "./models/4_0.pth", "./MLDS_hw2_1_data/pred_4.txt")