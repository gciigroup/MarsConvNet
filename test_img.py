import os
import matplotlib.pyplot as plt
from options import *
from models import *
from Normalize import *


# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# device = 'Automatic detection' if torch.cuda.is_available() else 'cpu'
results_path = './results_img/'
if not os.path.exists(results_path):
    os.mkdir(results_path)

path = './datas/npy/test/'
dataFile = os.path.join(path, 'datas2.npy')

data_list = np.load(dataFile, allow_pickle=True)
num = len(data_list)

model_dir = './snapshots/MC.pth'
ckp = torch.load(model_dir, map_location=opt.device)
net = MAN()
net = nn.DataParallel(net)
net = net.to(opt.device)
net.load_state_dict(ckp)
net.eval()

for i in range(num):
    df = data_list[i]
    df_data = df['data']
    p = df['label_p_arrive']
    s = df['label_s_arrive']
    label = df['label']
    d = df_data.shape[0]

    df_data2 = normalize(df_data)
    df_data2 = torch.from_numpy(df_data2)
    df_data2 = torch.permute(df_data2, [1, 0])
    df_data2 = torch.unsqueeze(df_data2, dim=0)
    df_data2 = df_data2.to(opt.device).type(torch.cuda.FloatTensor)
    x_d, x_p, x_s = net(df_data2)

    x_d = torch.squeeze(x_d, dim=0)
    x_d = torch.permute(x_d, [1, 0])
    x_d = x_d.cpu().detach().numpy()

    x_d = np.int64(x_d > 0.3)
    if np.max(x_d) == 0:
        pass
    else:
        r_d, c = np.where(x_d == np.max(x_d))
        x_d[r_d[0]:r_d[-1], :] = 1

    x_p = torch.squeeze(x_p, dim=0)
    x_p = torch.permute(x_p, [1, 0])
    x_p = x_p.cpu().detach().numpy()
    if np.max(x_p) < 0.1:
        x_p[:, :] = 0
    else:
        r_p, c = np.where(x_p == np.max(x_p))
        if r_p[0] - 20 < 0:
            x_p[int(r_p[0] + 20):, :] = 0
        elif r_p[0] + 20 > d:
            x_p[:int(r_p[0] - 20), :] = 0
        else:
            x_p[:int(r_p[0] - 20), :] = 0
            x_p[int(r_p[0] + 20):, :] = 0

    x_s = torch.squeeze(x_s, dim=0)
    x_s = torch.permute(x_s, [1, 0])
    x_s = x_s.cpu().detach().numpy()
    if np.max(x_s) < 0.1:
        x_s[:, :] = 0
    else:
        r_s, c = np.where(x_s == np.max(x_s))
        if r_s[0] - 20 < 0:
            x_s[int(r_s[0] + 20):, :] = 0
        elif r_s[0] + 20 > d:
            x_s[:int(r_s[0] - 20), :] = 0
        else:
            x_s[:int(r_s[0] - 20), :] = 0
            x_s[int(r_s[0] + 20):, :] = 0

    if df['label'] == 'EV':

        fig = plt.figure()
        ax = fig.add_subplot(411)
        plt.plot(df_data[:, 0], 'k')
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight': 'bold'}
        plt.tight_layout()
        ymin, ymax = ax.get_ylim()
        pl = plt.vlines(p, ymin, ymax, color='b', linewidth=2, label='P-arrival')
        sl = plt.vlines(s, ymin, ymax, color='r', linewidth=2, label='S-arrival')
        plt.legend(handles=[pl, sl], loc='upper right', borderaxespad=0., prop=legend_properties)
        # plt.ylabel('Amplitude counts', fontsize=12)
        ax.set_xticklabels([])

        ax = fig.add_subplot(412)
        plt.plot(df_data[:, 1], 'k')
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight': 'bold'}
        plt.tight_layout()
        ymin, ymax = ax.get_ylim()
        pl = plt.vlines(p, ymin, ymax, color='b', linewidth=2, label='P-arrival')
        sl = plt.vlines(s, ymin, ymax, color='r', linewidth=2, label='S-arrival')
        plt.legend(handles=[pl, sl], loc='upper right', borderaxespad=0., prop=legend_properties)
        # plt.ylabel('Amplitude counts', fontsize=12)
        ax.set_xticklabels([])

        ax = fig.add_subplot(413)
        plt.plot(df_data[:, 2], 'k')
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight': 'bold'}
        plt.tight_layout()
        ymin, ymax = ax.get_ylim()
        pl = plt.vlines(p, ymin, ymax, color='b', linewidth=2, label='P-arrival')
        sl = plt.vlines(s, ymin, ymax, color='r', linewidth=2, label='S-arrival')
        plt.legend(handles=[pl, sl], loc='upper right', borderaxespad=0., prop=legend_properties)
        # plt.ylabel('Amplitude counts', fontsize=12)
        ax.set_xticklabels([])

        ax = fig.add_subplot(414)
        plt.plot(x_d)
        plt.plot(x_p)
        plt.plot(x_s)
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight': 'bold'}
        plt.tight_layout()
        plt.savefig(results_path + '%d.png' % i, dpi=300)

    elif df['label'] == 'NO':

        fig = plt.figure()
        ax = fig.add_subplot(411)
        plt.plot(df_data[:, 0], 'k')
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight': 'bold'}
        plt.tight_layout()
        # plt.ylabel('Amplitude counts', fontsize=12)
        ax.set_xticklabels([])

        ax = fig.add_subplot(412)
        plt.plot(df_data[:, 1], 'k')
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight': 'bold'}
        plt.tight_layout()
        # plt.ylabel('Amplitude counts', fontsize=12)
        ax.set_xticklabels([])

        ax = fig.add_subplot(413)
        plt.plot(df_data[:, 2], 'k')
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight': 'bold'}
        plt.tight_layout()
        # plt.ylabel('Amplitude counts', fontsize=12)
        ax.set_xticklabels([])

        ax = fig.add_subplot(414)
        plt.plot(x_d)
        plt.plot(x_p)
        plt.plot(x_s)
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight': 'bold'}
        plt.tight_layout()
        plt.savefig(results_path + '%d.png' % i, dpi=300)
    
    print(f'\rstep :{i}/{num}', end='', flush=True)

