import os
import torch
import numpy as np
from torchvision import transforms
from .model import Backbone


class Arcface(object):
    def __init__(self):
        self.threshold = 0.8
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.get_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def get_model(self):
        path = os.path.join(os.path.dirname(__file__), 'weight')
        model = Backbone(50, 0.6, 'ir_se')
        model.load_state_dict(torch.load(path+'/model_final.pth'))
        model = model.to(self.device).eval()
        return model

    def __call__(self, face):
        face_tensor = self.transform(face).to(self.device).unsqueeze(0)
        emb = self.model(face_tensor)
        return emb.detach().cpu().numpy()

    # def get_emd(self, face):
    #     face_tensor = self.transform(face).to(self.device).unsqueeze(0)
    #     emb = self.model(face_tensor)
    #     return emb
    #
    # def infer(self, face):
    #     emd = self.get_emd(face)
    #     with torch.no_grad():
    #         diff = emd.unsqueeze(-1) - self.targets
    #         dist = torch.sum(torch.pow(diff, 2), dim=1)
    #         minimum, min_idx = torch.min(dist, dim=1)
    #         min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
    #         min_idx = min_idx.cpu().numpy()
    #         minimum = minimum.cpu().numpy()
    #
    #     return min_idx, minimum
    #
    # def infer_multi(self, faces):
    #     embs = []
    #     for face in faces:
    #         emd = self.get_emd(face)
    #         embs.append(emd)
    #     source_embs = torch.cat(embs)
    #
    #     with torch.no_grad():
    #         diff = source_embs.unsqueeze(-1) - self.targets
    #         dist = torch.sum(torch.pow(diff, 2), dim=1)
    #         minimum, min_idx = torch.min(dist, dim=1)
    #         min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
    #         min_idx = min_idx.cpu().numpy()
    #         minimum = minimum.cpu().numpy()
    #
    #     return min_idx, minimum
    #
    # def __load_facebank(self):
    #     path = os.path.join(os.path.dirname(__file__), 'facebank')
    #     embeddings = torch.load(path+'/facebank.pth').transpose(1, 0).unsqueeze(0)
    #     names = np.load(path+'/names.npy')
    #     return embeddings, names

if __name__ =="__main__":
    arcface = Arcface()
