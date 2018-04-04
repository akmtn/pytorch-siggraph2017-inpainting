import cv2
import numpy as np


def tensor2cvimg(src):
    '''return np.array
        uint8
        [0, 255]
        BGR
        (H, W, C)
    '''
    out = src.copy() * 255
    out = out.transpose((1, 2, 0)).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    return out

def cvimg2tensor(src):
    out = src.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out = out.transpose((2,0,1)).astype(np.float64)
    out = out / 255

    return out
