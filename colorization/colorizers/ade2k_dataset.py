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
            return 20210
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
        img = img.convert("RGB")
        mask = Image.open(mask_path)

        img_l_orig, img_l_rs, img_ab_rs = preprocess_img(np.asarray(img), HW=(256,256), resample=Image.NEAREST, ade2k=True)
        mask_rs = mask.resize((256, 256), resample=Image.NEAREST)

        # need to output 3 things (2 things for now)
        # 1. ade2k image bw resized
        # 2. ade2k mask resized, ignore for now
        # 3. ade2k 1 x 2 x 256 x 256 image ab tensor
        return img_l_rs, img_ab_rs

if __name__=="__main__":
    ab_norm = 110.
    ab_max = 110.
    ab_quant = 10.
    A = 2 * ab_max / ab_quant + 1

    # print(sys.path)
    train_dataset = ADE2kDataset("../data/ADEChallengeData2016/images/training", \
                            "../data/ADEChallengeData2016/annotations/training", \
                            "train")
    val_dataset = ADE2kDataset("../data/ADEChallengeData2016/images/validation", \
                            "../data/ADEChallengeData2016/annotations/validation", \
                            "val")

    model = siggraph17(pretrained=False)
    
    # Params
    model_dir = "model_weights/"
    num_epochs = 10
    learning_rate = 0.001
    batch = 16

    print("checkpt 1")

    # Moving model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.cuda()

    print("checkpt 2")

    # Dataset to dataloader
    trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size = batch, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size = batch, pin_memory=True)

    print("checkpt 3")

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("checkpt 4")

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Starting epoch", epoch)
        running_loss = 0.0
        epoch_loss = 0.0
        epoch_path = model_dir + "epoch_" + str(epoch) + ".pt"
        model.train()
        if os.path.exists(epoch_path):
            print("exists! loading from ", epoch_path)
            model.load_state_dict(torch.load(epoch_path))
        else:
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                l_inputs, ab_inputs = data
                
                # print("input size after data", list(inputs.size()))
                # flattening input and label due to batch size = 1 (dataloader adds dim for batch)
                l_inputs = torch.squeeze(l_inputs, 1)
                ab_inputs = torch.squeeze(ab_inputs, 1)
                # print("input size after squeeze", list(inputs.size()))

                l_inputs = l_inputs.to(device)
                # print("input size after moving", list(inputs.size()))
                ab_inputs = ab_inputs.to(device)

                print("lshape", l_inputs.shape)
                print("abshape", ab_inputs.shape)
                ab_rs = ab_inputs[:, :, ::4, ::4]

                # zero the parameter gradients
                optimizer.zero_grad()
                # print("input size after opt", list(inputs.size()))

                # forward + backward + optimize

                # print("my input is size", list(inputs.size()))
                class_outputs = model(l_inputs)
                ab_enc = encode_ab_ind(ab_rs, ab_max, ab_quant, A)
                loss = 0
                # for i in range(batch):
                #     print("shape1", class_outputs.type(torch.FloatTensor)[i, :, :, :].shape)
                #     print("shape2", ab_enc.type(torch.LongTensor)[i, :, :].shape)
                #     loss += criterion(class_outputs.type(torch.FloatTensor)[i, :, :, :], ab_enc.type(torch.LongTensor)[i, :, :])
                print("shape1", class_outputs.type(torch.FloatTensor).shape)
                print("shape2", ab_enc[:, 0, :, :].type(torch.LongTensor).shape)
                if torch.cuda.is_available():
                    loss = criterion(class_outputs.type(torch.cuda.FloatTensor), ab_enc[:, 0, :, :].type(torch.cuda.LongTensor))
                else:
                    loss += criterion(class_outputs.type(torch.FloatTensor), ab_enc[:, 0, :, :].type(torch.LongTensor))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 6 == 0:    # print every 6 batches of batch = 16, 96 samples
                    print('\t[%d, %5d] loss: %.3f' %
                        (epoch + 1, i*batch, running_loss / 6))
                    running_loss = 0.0


            print("Epoch", epoch, "complete, saving to...", epoch_path)
            torch.save(model.state_dict(), epoch_path)
            epoch_loss_avg = epoch_loss/(20210/batch)
            print("Saved epoch", epoch, "avg epoch loss = ", epoch_loss_avg)

        print("Running validation...")
        model.eval()
        val_loss = 0.0
        running_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # print("input size after data", list(inputs.size()))
                # flattening input and label due to batch size = 1 (dataloader adds dim for batch)
                inputs = torch.squeeze(inputs, 1)
                labels = torch.squeeze(labels, 1)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # only run forward for val
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # print statistics
                running_val_loss += loss.item()
                val_loss += loss.item()
                if i % 6 == 0:    # print every 6 batches of batch = 16, 96 samples
                    print('\t[%d, %5d] loss: %.3f' %
                        (epoch + 1, i*batch, running_val_loss / 6))
                    running_val_loss = 0.0

        print("Epoch", epoch, "validation loss:", val_loss/(2000/batch))
        with open("model_weights/val_loss_" + str(epoch) +".txt", "w") as f:
            f.write("Epoch " + str(epoch) + " validation loss: " + str(val_loss/(2000/batch)))
            f.close()


    print('Finished Training')
   


        