from .model import PNet, RNet, ONet
import numpy as np
import math
import torch
import cv2


class MTCNN(object):
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.__device = 'cuda'
            self.__tensor = torch.cuda.FloatTensor
        else:
            self.__device = 'cpu'
            self.__tensor = torch.FloatTensor
        self._pnet = PNet().to(self.__device).eval()
        self._rnet = RNet().to(self.__device).eval()
        self._onet = ONet().to(self.__device).eval()
        self.scales = [0.3, 0.15, 0.07, 0.035]
        self.thresholds = [0.7, 0.8, 0.9]
        self.nms_thresholds = [0.7, 0.7, 0.7]

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = []
        for s in self.scales:  # run P-Net on different scales
            boxes = self._run_pnet(image, scale=s)
            bbox.append(boxes)
        bbox = [i for i in bbox if i is not None]
        bbox = np.vstack(bbox)
        bbox = self._bbox_filter(bbox[:, :5], offsets=bbox[:, 5:], nms_threshold=self.nms_thresholds[0])
        bbox, offsets = self._run_rnet(bbox, image)
        bbox = self._bbox_filter(bbox, offsets, nms_threshold=self.nms_thresholds[0])
        bbox, landmarks = self._run_onet(bbox, image)
        return landmarks, bbox

    def _run_pnet(self, image, scale):
        height, width, _ = image.shape
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = cv2.resize(image, (sw, sh))
        img = np.asarray(img, 'float32')
        img = self.__tensor(self._preprocess(img))

        output = self._pnet(img)
        probs = output[1].data.cpu().numpy()[0, 1, :, :]
        offsets = output[0].data.cpu().numpy()

        boxes = self._generate_bboxes(probs, offsets, scale, threshold=self.thresholds[0])
        if len(boxes) == 0:
            return None

        keep = self._nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]

    def _run_rnet(self, bbox, image, size=24):
        img_boxes = self._get_image_boxes(bbox, image, size=size)
        img_boxes = self.__tensor(img_boxes)
        output = self._rnet(img_boxes)
        offsets = output[0].data.cpu().numpy()  # shape [n_boxes, 4]
        probs = output[1].data.cpu().numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[1])[0]
        bbox = bbox[keep]
        bbox[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        return bbox, offsets

    def _run_onet(self, bbox, image):
        img_boxes = self._get_image_boxes(bbox, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = self.__tensor(img_boxes)
        output = self._onet(img_boxes)
        landmarks = output[0].data.cpu().numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.cpu().numpy()  # shape [n_boxes, 4]
        probs = output[2].data.cpu().numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[2])[0]
        bbox = bbox[keep]
        bbox[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bbox[:, 2] - bbox[:, 0] + 1.0
        height = bbox[:, 3] - bbox[:, 1] + 1.0
        xmin, ymin = bbox[:, 0], bbox[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bbox = self._calibrate_box(bbox, offsets)
        keep = self._nms(bbox, self.nms_thresholds[2], mode='min')
        bbox = bbox[keep]
        landmarks = landmarks[keep]

        return bbox, landmarks

    def _preprocess(self, img):
        """Preprocessing step before feeding the network. """
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = (img - 127.5) * 0.0078125
        return img

    def _generate_bboxes(self, probs, offsets, scale, threshold):
        """
           Generate bounding boxes at places where there is probably a face.
        """
        stride = 2
        cell_size = 12

        inds = np.where(probs > threshold)

        if inds[0].size == 0:
            return np.array([])

        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # P-Net is applied to scaled images, so we need to rescale bounding boxes back
        bounding_boxes = np.vstack([
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score, offsets
        ])

        return bounding_boxes.T

    def _nms(self, boxes, overlap_threshold=0.5, mode='union'):
        """ Pure Python NMS baseline. """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            if mode is 'min':
                ovr = inter / np.minimum(areas[i], areas[order[1:]])
            else:
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= overlap_threshold)[0]
            order = order[inds + 1]

        return keep

    def _calibrate_box(self, bboxes, offsets):
        """Transform bounding boxes to be more like true bounding boxes.
        'offsets' is one of the outputs of the nets.
        """
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        w = np.expand_dims(w, 1)
        h = np.expand_dims(h, 1)

        translation = np.hstack([w, h, w, h]) * offsets
        bboxes[:, 0:4] = bboxes[:, 0:4] + translation
        return bboxes

    def _convert_to_square(self, bboxes):
        """
            Convert bounding boxes to a square form.
        """
        square_bboxes = np.zeros_like(bboxes)
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        h = y2 - y1 + 1.0
        w = x2 - x1 + 1.0
        max_side = np.maximum(h, w)
        square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
        square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
        square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
        square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
        return square_bboxes

    def _bbox_filter(self, bbox, offsets, nms_threshold):
        keep = self._nms(bbox, nms_threshold)
        bbox = bbox[keep]
        offsets = offsets[keep]
        bbox = self._calibrate_box(bbox, offsets)
        bbox = self._convert_to_square(bbox)
        bbox[:, 0:4] = np.round(bbox[:, 0:4])
        return bbox

    def _correct_bboxes(self, bboxes, width, height):
        """Crop boxes that are too big and get coordinates
        with respect to cutouts.
        """
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
        num_boxes = bboxes.shape[0]

        x, y, ex, ey = x1, y1, x2, y2
        dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
        edx, edy = w.copy() - 1.0, h.copy() - 1.0

        ind = np.where(ex > width - 1.0)[0]
        edx[ind] = w[ind] + width - 2.0 - ex[ind]
        ex[ind] = width - 1.0

        ind = np.where(ey > height - 1.0)[0]
        edy[ind] = h[ind] + height - 2.0 - ey[ind]
        ey[ind] = height - 1.0

        ind = np.where(x < 0.0)[0]
        dx[ind] = 0.0 - x[ind]
        x[ind] = 0.0

        ind = np.where(y < 0.0)[0]
        dy[ind] = 0.0 - y[ind]
        y[ind] = 0.0
        return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
        return_list = [i.astype('int32') for i in return_list]

        return return_list

    def _get_image_boxes(self, bounding_boxes, img, size):
        """Cut out boxes from the image.
        """
        num_boxes = len(bounding_boxes)
        height, width, _ = img.shape

        [dy, edy, dx, edx, y, ey, x, ex, w, h] = self._correct_bboxes(bounding_boxes, width, height)
        img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

        for i in range(num_boxes):
            img_box = np.zeros((h[i], w[i], 3), 'uint8')

            img_array = np.asarray(img, 'uint8')
            img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
                img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

            img_box = cv2.resize(img_box, (size, size))
            img_box = np.asarray(img_box, 'float32')

            img_boxes[i, :, :, :] = self._preprocess(img_box)

        return img_boxes


if __name__ == "__main__":
    mtcnn = MTCNN()
