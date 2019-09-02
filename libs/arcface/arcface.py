import os
import torch
import numpy as np
from torchvision import transforms
from .model import Backbone


class Arcface(object):
    def __init__(self):
        self.threshold = 0.8
        self.__cuda = torch.cuda.is_available()
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.get_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def get_model(self):
        path = os.path.join(os.path.dirname(__file__), 'weight', 'model_final.pth')
        model = Backbone(50, 0.6, 'ir_se')
        if self.__cuda:
            param = torch.load(path)
            model.load_state_dict(param)
            model = model.cuda()
        else:
            param = torch.load(path, map_location=lambda storage, loc: storage)
            model.load_state_dict(param)
        model = model.eval()
        return model

    def __call__(self, face):
        face_tensor = self.transform(face).to(self.__device).unsqueeze(0)
        emb = self.model(face_tensor)
        return emb.detach().cpu().numpy()


if __name__ =="__main__":
    arcface = Arcface()
