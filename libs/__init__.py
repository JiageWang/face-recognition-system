from libs.utils.align import warp_and_crop_face, get_reference_facial_points, get_aligned_faces
from .arcface.arcface import Arcface
from .mtcnn.mtcnn import MTCNN


class FaceModel(object):
    def __init__(self, face_bank):
        self.mtcnn = MTCNN()
        self.arcface = Arcface()
        self.threshold = 0.8
        self.facebank = None
        self.ids = None
        self.facebank = face_bank

    def __call__(self, img):
        landmarks, bboxs = self.mtcnn(img)
        faces = get_aligned_faces(img, bboxs, landmarks)
        embeddings = []
        for face in faces:
            embedding = self.arcface(face)
            embeddings.append(embedding)
        return landmarks, bboxs, faces, embeddings
