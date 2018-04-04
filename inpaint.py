import argparse
import os
import torch
from torch.legacy import nn
from torch.legacy.nn.Sequential import Sequential
import cv2
import numpy as np
from torch.utils.serialization import load_lua
import torchvision.utils as vutils
from utils import *
from poissonblending import prepare_mask, blend


parser = argparse.ArgumentParser()
parser.add_argument('--input', default='none', help='Input image')
parser.add_argument('--mask', default='none', help='Mask image')
parser.add_argument('--model_path', default='completionnet_places2.t7', help='Trained model')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='use GPU')
parser.add_argument('--postproc', default=False, action='store_true',
                    help='Disable post-processing')
opt = parser.parse_args()
print(opt)


# load Completion Network
data = load_lua(opt.model_path)
model = data.model
model.evaluate()
datamean = data.mean

# load data
input_img = cv2.imread(opt.input)
I = torch.from_numpy(cvimg2tensor(input_img)).float()

if opt.mask != 'none':
    input_mask = cv2.imread(opt.mask)
    M = torch.from_numpy(
                cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY) / 255).float()
    M[M <= 0.2] = 0.0
    M[M > 0.2] = 1.0
    M = M.view(1, M.size(0), M.size(1))
    assert I.size(1) == M.size(1) and I.size(2) == M.size(2)

else:
    # generate random holes
    M = torch.FloatTensor(1, I.size(1), I.size(2)).fill_(0)
    nHoles = np.random.randint(1, 4)
    print(nHoles)
    print('w: ', I.size(2))
    print('h: ', I.size(1))
    for _ in range(nHoles):
        mask_w = np.random.randint(32, 128)
        mask_h = np.random.randint(32, 128)
        assert I.size(1) > mask_h or I.size(2) > mask_w
        px = np.random.randint(0, I.size(2)-mask_w)
        py = np.random.randint(0, I.size(1)-mask_h)
        M[:, py:py+mask_h, px:px+mask_w] = 1


for i in range(3):
    I[i, :, :] = I[i, :, :] - datamean[i]

# make mask_3ch
M_3ch = torch.cat((M, M, M), 0)

Im = I * (M_3ch*(-1)+1)

# set up input
input = torch.cat((Im, M), 0)
input = input.view(1, input.size(0), input.size(1), input.size(2)).float()

if opt.gpu:
    print('using GPU...')
    model.cuda()
    input = input.cuda()

# evaluate
res = model.forward(input)[0].cpu()

# make out
for i in range(3):
    I[i, :, :] = I[i, :, :] + datamean[i]

out = res.float()*M_3ch.float() + I.float()*(M_3ch*(-1)+1).float()

# post-processing
if opt.postproc:
    print('post-postprocessing...')
    target = input_img    # background
    source = tensor2cvimg(out.numpy())    # foreground
    mask = input_mask
    out = blend(target, source, mask, offset=(0, 0))

    out = torch.from_numpy(cvimg2tensor(out))


# save images
print('save images...')
vutils.save_image(out, 'out.png', normalize=True)
# vutils.save_image(Im, 'masked_input.png', normalize=True)
# vutils.save_image(M_3ch, 'mask.png', normalize=True)
# vutils.save_image(res, 'res.png', normalize=True)
print('Done')
