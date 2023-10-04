import argparse
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--train_path', type=str, default='./datas/npy/train/')
parser.add_argument('--test_path', type=str, default='./datas/npy/test/')
parser.add_argument('--valid_path', type=str, default='./datas/npy/valid/')
parser.add_argument('--image_path', type=str, default='./image/')
parser.add_argument('--loss_file', type=str, default='./loss/')
parser.add_argument('--evals_file', type=str, default='./evals/')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--snapshots_folder', type=str, default='./snapshots/')
parser.add_argument('--device', type=str, default='Automatic detection')
parser.add_argument('--norm_mode', type=str, default='std')
parser.add_argument('--use_stft', action='store_true', help='no lr cos schedule')

opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
