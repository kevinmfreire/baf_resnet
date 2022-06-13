import os
import argparse
#import imageio
import numpy as np
# import matplotlib.pyplot as plt
import time

# Importing torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# For MNIST dataset and visualization
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from loss import CompoundLoss
from network import bafnet
from loader import get_loader
from tqdm import tqdm


parser= argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--load_mode', type=int, default=1)
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--saved_path', type=str, default='./processed_data/npy_img/')
parser.add_argument('--save_path', type=str, default='./model/')
# parser.add_argument('--save_path', type=str, default='../NCSSMP/')
parser.add_argument('--test_patient', type=str, default='L058')

parser.add_argument('--save_iters', type=int, default=50)
parser.add_argument('--print_iters', type=int, default=50)
parser.add_argument('--decay_iters', type=int, default=6000)
parser.add_argument('--gan_alt', type=int, default=2)

parser.add_argument('--transform', type=bool, default=False)
# if patch training, batch size is (--patch_n * --batch_size)
parser.add_argument('--patch_n', type=int, default=16)		# default = 4
parser.add_argument('--patch_size', type=int, default=32)	# default = 100
parser.add_argument('--batch_size', type=int, default=4)	# default = 5
parser.add_argument('--image_size', type=int, default=512)

parser.add_argument('--lr', type=float, default=0.0002) # 5e-5 without decaying rate
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--load_chkpt', type=bool, default=False)

parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3072.0)

args = parser.parse_args()

data_loader = get_loader(mode=args.mode,
							load_mode=args.load_mode,
							saved_path=args.saved_path,
							test_patient=args.test_patient,
							patch_n=(args.patch_n if args.mode=='train' else None),
							patch_size=(args.patch_size if args.mode=='train' else None),
							transform=args.transform,
							batch_size=(args.batch_size if args.mode=='train' else 1),
							num_workers=args.num_workers)

# Check CUDA's presence
cuda_is_present = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda_is_present else torch.FloatTensor

image_size = args.image_size if args.patch_size == None else args.patch_size

def to_cuda(data):
	return data.cuda() if cuda_is_present else data

if args.load_chkpt:
	print('Loading Chekpoint')
	whole_model = torch.load(args.save_path+ 'epoch_15_ckpt.pth.tar', map_location=torch.device('cuda' if cuda_is_present else 'cpu'))
	net_state_dict,opt_state_dict = whole_model['net_state_dict'], whole_model['opt_state_dict']
	net = bafnet()
	net = to_cuda(net)
	optimizer_net = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.01,0.99))
	net.load_state_dict(net_state_dict)
	optimizer_net.load_state_dict(opt_state_dict)
	cur_epoch = whole_model['epoch']
	total_iters = whole_model['total_iters']
	lr = whole_model['lr']
	# g_net = torch.nn.DataParallel(g_net, device_ids=[0, 1])
	# d_net = torch.nn.DataParallel(d_net, device_ids=[0, 1])
	print('Current Epoch:{}, Total Iters: {}, Learning rate: {}, Batch size: {}'.format(cur_epoch, total_iters, lr, args.batch_size))
else:
	print('Training model from scrath')
	net = bafnet()
	net = to_cuda(net)
	optimizer_net = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.01,0.999))
	cur_epoch = 0
	total_iters = 0
	lr = args.lr

# Losses
criterion = CompoundLoss()
# Dloss = to_cuda(Dloss)
# criterion = to_cuda(criterion)

losses = []
start_time = time.time()
tq_epoch = tqdm(range(cur_epoch, args.num_epochs),position=1, leave=True, desc='Epochs')
torch.autograd.set_detect_anomaly(True)
for epoch in tq_epoch:

    # Initializing sum of losses for discriminator and generator
    loss_sum, count = 0, 0

    net.train()

    data_tqdm = tqdm(data_loader, position=0, leave=True, desc='Iters')
    for i, (x, y) in enumerate(data_tqdm):
        total_iters += 1
        count += 1
        shape_ = x.shape[-1]

        # add 1 channel
        x = x.unsqueeze(0).float()
        y = y.unsqueeze(0).float()

        # If patch training
        if args.patch_size:
            x = x.view(-1, 1, args.patch_size, args.patch_size)
            y = y.view(-1, 1, args.patch_size, args.patch_size)

        # If batch training without any patch size
        if args.batch_size and args.patch_size == None:
            x = x.view(-1, 1, shape_, shape_)
            y = y.view(-1, 1, shape_, shape_)

        y = to_cuda(y)
        x = to_cuda(x)

		# Predictions
        pred = net(x)

		# Training generator
        optimizer_net.zero_grad()
        net.zero_grad()
        loss = criterion(pred, y)
        loss.backward(retain_graph=True)
        optimizer_net.step()

        loss_sum += loss.detach().item()
		
        data_tqdm.set_postfix({'ITER': i+1, 'LOSS': '{:.5f}'.format(loss.item())})

		# Saving model after every 10 iterations
        if total_iters % args.save_iters == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
                print('Create path : {}'.format(args.save_path))
            print('Saving model to: ' + args.save_path)
            saved_model = {
                'epoch': epoch ,
                'net_state_dict': net.state_dict(),
                'opt_state_dict': optimizer_net.state_dict(),
                'lr': lr,
                'total_iters': total_iters
            }
            torch.save(saved_model, '{}latest_ckpt.pth.tar'.format(args.save_path))
	
	# Calculating average loss
    avg_loss = loss_sum/float(count)
    losses.append(avg_loss)

	# Saving to google drive
    # save_loss = '/gdrive/MyDrive/rigan_model/loss_arr.npy'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))
    print('Saving model to: ' + args.save_path)
    saved_model = {
                'epoch': epoch ,
                'net_state_dict': net.state_dict(),
                'opt_state_dict': optimizer_net.state_dict(),
                'lr': lr,
                'total_iters': total_iters
            }
    torch.save(saved_model, '{}latest_ckpt.pth.tar'.format(args.save_path))

    save_loss = './model/loss_arr.py'
    np.save(save_loss, losses, allow_pickle=True)

    tq_epoch.set_postfix({'STEP': total_iters,'AVG_LOSS': '{:.5f}'.format(avg_loss)})
	
	# Saving model after every 10 epoch
    # if epoch % 2 == 0:
    #     saved_model = {
    #         'epoch': epoch ,
    #         'netG_state_dict': net.state_dict(),
    #         'optG_state_dict': optimizer_net.state_dict(),
    #         'lr': lr,
    #         'total_iters': total_iters
    #     }
    #     torch.save(saved_model, '{}latest_ckpt.pth.tar'.format(args.save_path))
    #     cmd1 = 'cp {}latest_ckpt.pth.tar /gdrive/MyDrive/rigan_model/epoch_{}_ckpt.pth.tar'.format(args.save_path, epoch)
    #     cmd2 = 'cp {}latest_ckpt.pth.tar /gdrive/MyDrive/rigan_model/'.format(args.save_path)
    #     os.system(cmd1)
    #     os.system(cmd2)