import torch
from torch import nn
from torch.optim import Adam
import os


class MarioModel(nn.Module):
    def __init__(self, action_size: int):
        super(MarioModel, self).__init__()
        self.__input_dim = [1, 1, 13, 16]
        channel, height, width = self.__input_dim[1:]

        # 1層目(conv -> relu -> maxpool)
        self.h_conv1 = nn.Conv2d(in_channels=channel, out_channels=16, kernel_size=[7, 9], padding_mode='replicate')
        self.relu1 = nn.ReLU(inplace=True)
        self.h_pool1 = nn.MaxPool2d(kernel_size=[1, 2], stride=[2, 2])

        # 2層目(conv -> relu -> maxpool)
        self.h_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[1, 1])
        self.relu2 = nn.ReLU(inplace=True)
        self.h_pool2 = nn.MaxPool2d(kernel_size=[1, 2], stride=[2, 2])

        # 3層目(conv -> relu -> maxpool)
        self.h_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[1, 1])
        self.relu3 = nn.ReLU(inplace=True)
        self.h_pool3 = nn.MaxPool2d(kernel_size=[1, 2], stride=[2, 2])

        # 4層目(conv -> relu -> maxpool)
        self.h_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1])
        self.relu4 = nn.ReLU(inplace=True)
        self.h_pool4 = nn.MaxPool2d(kernel_size=[1, 1])

        # 5層目(conv -> relu -> maxpool)
        self.h_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1])
        self.relu5 = nn.ReLU(inplace=True)
        self.h_pool5 = nn.MaxPool2d(kernel_size=[1, 1])

        # 6層目(conv -> relu -> maxpool)
        self.h_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1])
        self.relu6 = nn.ReLU(inplace=True)
        self.h_pool6 = nn.MaxPool2d(kernel_size=[1, 1])

        # 出力層(reshape -> fully_connect)
        self.out = nn.Linear(in_features=64, out_features=action_size)

        return

    def forward(self, screen: torch.Tensor = torch.zeros([1, 1, 13, 16])):
        # 1層目
        z1 = self.relu1(self.h_conv1(screen))  # [1, 16, 7, 8]
        y1 = self.h_pool1(z1)  # [1, 16, 4, 4]

        # 2層目
        z2 = self.relu2(self.h_conv2(y1))  # [1, 32, 4, 4]
        y2 = self.h_pool2(z2)  # [1, 32, 2, 2]

        # 3層目
        z3 = self.relu3(self.h_conv3(y2))  # [1, 64, 2, 2]
        y3 = self.h_pool3(z3)  # [1, 64, 1, 1]

        # 4層目
        z4 = self.relu4(self.h_conv4(y3))  # [1, 64, 1, 1]
        y4 = self.h_pool4(z4)  # [1, 64, 1, 1]

        # 5層目
        z5 = self.relu5(self.h_conv5(y4))  # [1, 64, 1, 1]
        y5 = self.h_pool5(z5)  # [1, 64, 1, 1]

        # 6層目
        z6 = self.relu6(self.h_conv6(y5))  # [1, 64, 1, 1]
        y6 = self.h_pool6(z6)  # [1, 64, 1, 1]

        # 出力層
        flat = torch.reshape(y6, (-1, 64))  # [1, 64]
        output = self.out(flat)  # [1, 2]

        return output

    def predict(self, screen: torch.Tensor):
        """
        Returns
            prediction: numpy.ndarray  shape=(action_size, )
        """
        output = self.forward(screen)
        prediction = output.detach().numpy()[0]

        return prediction

    def load(self, path: str = "./pth_models/mario.pth"):
        # モデルのパラメータを読み込む
        # GPUが有効ならばGPUを使う
        self.load_state_dict(torch.load(path))
        if torch.cuda.is_available():
            self.to('cuda')
            pass
        print("Model is loaded from: {}".format(path))

        return

    def save(self, directory: str = "./pth_models", filename: str = "mario.pth"):
        # モデルを保存する
        # 保存時はCPUに変換しておく
        os.makedirs(directory, exist_ok=True)
        path = "{0}/{1}".format(directory.rstrip('/'), filename)

        torch.save(self.to('cpu').state_dict(), path)
        print("Model is saved to: {}".format(path))

        return

    def get_optimizer(self, lr=1e-4):
        return Adam(self.parameters(), 1e-4)

    def get_loss_function(self):
        return nn.MSELoss()
