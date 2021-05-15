import torch
import os, sys
from PIL import Image
from skimage import color
from colorization.colorizers.util import *
from colorization.colorizers.siggraph17 import siggraph17

import torch.optim as optim

class ADE2kDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, split):
        """
        Pass in the directory of images and masks
        train = True if train, false if val
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.split = split
    
    def __len__(self):
        if self.split == "train":
            return 20000
        elif self.split == "val":
            return 2000
        else:
            return 0

    def __getitem__(self, index):
        """
        Convert index (0 indexed) into filename (1 indexed)
        """
        file_prefix = "ADE_"
        if self.split == "train":
            file_prefix += "train_"
        elif self.split == "val":
            file_prefix += "val_"
        else:
            print("invalid split")
            return
        
        # files are named in 1-indexed
        index += 1
        num_zeros = 8 - len(str(index))
        file_prefix += "0"*num_zeros + str(index)

        img_path = self.img_dir + "/" + file_prefix + ".jpg"
        mask_path = self.mask_dir + "/" + file_prefix + ".png"

        # print("img path", img_path)
        # print("mask path", mask_path)

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img_l_orig, img_l_rs, img_ab_rs = preprocess_img(np.asarray(img), HW=(256,256), resample=Image.NEAREST, ade2k=True)
        mask_rs = mask.resize((256, 256), resample=Image.NEAREST)

        # need to output 3 things (2 things for now)
        # 1. ade2k image bw resized
        # 2. ade2k mask resized, ignore for now
        # 3. ade2k 1 x 2 x 256 x 256 image ab tensor
        return img_l_rs, img_ab_rs
        


if __name__=="__main__":
    # print(sys.path)
    train_dataset = ADE2kDataset("/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/images/training", \
                            "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/annotations/training", \
                            "train")
    val_dataset = ADE2kDataset("/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/images/validation", \
                            "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/annotations/validation", \
                            "val")

    model = siggraph17(pretrained=False)
    
    # Params
    model_dir = "model_weights/"
    num_epochs = 10
    learning_rate = 0.001

    print("checkpt 1")

    # Moving model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.cuda()

    print("checkpt 2")

    # Dataset to dataloader
    trainloader = torch.utils.data.DataLoader(train_dataset, pin_memory=True)
    valloader = torch.utils.data.DataLoader(train_dataset, pin_memory=True)

    print("checkpt 3")

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("checkpt 4")

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Starting epoch", epoch)
        running_loss = 0.0
        epoch_loss = 0.0
        epoch_path = model_dir + "epoch_" + str(epoch) + ".pt"
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # print("input size after data", list(inputs.size()))
            # flattening input and label due to batch size = 1 (dataloader adds dim for batch)
            inputs = torch.squeeze(inputs, 0)
            labels = torch.squeeze(labels, 0)
            # print("input size after squeeze", list(inputs.size()))


            inputs = inputs.to(device)
            # print("input size after moving", list(inputs.size()))
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # print("input size after opt", list(inputs.size()))

            # forward + backward + optimize

            # print("my input is size", list(inputs.size()))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                print('\t[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        print("Epoch", epoch, "complete, saving to...", epoch_path)
        torch.save(model.state_dict(), epoch_path)
        epoch_loss_avg = epoch_loss/20000
        print("Saved epoch", epoch, "avg epoch loss = ", epoch_loss_avg)

        print("Running validation...")
        val_loss = 0.0
        running_val_loss = 0.0
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # print("input size after data", list(inputs.size()))
            # flattening input and label due to batch size = 1 (dataloader adds dim for batch)
            inputs = torch.squeeze(inputs, 0)
            labels = torch.squeeze(labels, 0)


            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # only run forward for val
            outputs = model(inputs)

            # print statistics
            running_val_loss += loss.item()
            val_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                print('\t[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_val_loss / 100))
                running_val_loss = 0.0

        print("Epoch", epoch, "validation loss:", val_loss)


    print('Finished Training')
   


        