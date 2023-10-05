import os
import matplotlib.pyplot as plt
from options import *
from models import *
from Normalize import *
from metrics import *


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

pre_ds = []
rec_ds = []
fsc_ds = []

pre_ps = []
rec_ps = []
fsc_ps = []

pre_ss = []
rec_ss = []
fsc_ss = []
for i in range(num):
    df = data_list[i]
    df_data = df['data']
    p = df['label_p_arrive']
    s = df['label_s_arrive']
    label = df['label']
    d = df_data.shape[0]

    label_d = np.zeros([1, d])
    label_p = np.zeros([1, d])
    label_s = np.zeros([1, d])

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

        x_d_c = x_d.copy()
        x_p_c = x_p.copy()
        x_s_c = x_s.copy()

        sd = s - p
        if int(s) + int(1.4 * sd) <= d:
            label_d[:, int(p):int(s + (1.4 * sd))] = 1
        else:
            label_d[:, int(p):d] = 1

        if p and (p - 20 >= 0) and (p + 20 < d):
            label_p[:, int(p - 20):int(p + 20)] = np.exp(
                -(np.arange(int(p - 20), int(p + 20)) - p) ** 2 / (2 * (10) ** 2))[:int(p + 20) - int(p - 20)]
        elif p and (p + 20 >= d):
            label_p[:, int(p - 20): d] = np.exp(
                -(np.arange(int(p - 20), int(p + 20)) - p) ** 2 / (2 * (10) ** 2))[:(d - int(p - 20))]
        elif p and (p - 20 < 0):
            label_p[:, 0: int(p + 20)] = np.exp(
                -(np.arange(int(p - 20), int(p + 20)) - p) ** 2 / (2 * (10) ** 2))[int(20 - p):]

        if s and (s - 20 >= 0) and (s + 20 < d):
            label_s[:, int(s - 20): int(s + 20)] = np.exp(
                -(np.arange(int(s - 20), int(s + 20)) - s) ** 2 / (2 * (10) ** 2))[:int(s + 20) - int(s - 20)]
        elif s and (s + 20 >= d):
            label_s[:, int(s - 20): d] = np.exp(
                -(np.arange(int(s - 20), int(s + 20)) - s) ** 2 / (2 * (10) ** 2))[:(d - int(s - 20))]
        elif s and (s - 20 < 0):
            label_s[:, 0: int(s + 20)] = np.exp(
                -(np.arange(int(s - 20), int(s + 20)) - s) ** 2 / (2 * (10) ** 2))[int(20 - s):]

        label_d = label_d.transpose((1, 0))
        label_p = label_p.transpose((1, 0))
        label_s = label_s.transpose((1, 0))

        pre_d, rec_d, fsc_d = evalue(x_d_c, label_d, label)
        print('Precision_d:', pre_d)
        print('Recall_d:', rec_d)
        print('F-score_d:', fsc_d)
        pre_ds.append(pre_d)
        rec_ds.append(rec_d)
        fsc_ds.append(fsc_d)

        pre_p, rec_p, fsc_p = evalue(x_p_c, label_p, label)
        print('Precision_p:', pre_p)
        print('Recall_p:', rec_p)
        print('F-score_p:', fsc_p)
        pre_ps.append(pre_p)
        rec_ps.append(rec_p)
        fsc_ps.append(fsc_p)

        pre_s, rec_s, fsc_s = evalue(x_s_c, label_s, label)
        print('Precision_s:', pre_s)
        print('Recall_s:', rec_s)
        print('F-score_s:', fsc_s)
        pre_ss.append(pre_s)
        rec_ss.append(rec_s)
        fsc_ss.append(fsc_s)

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

        x_d_c = x_d.copy()
        x_p_c = x_p.copy()
        x_s_c = x_s.copy()

        label_d = label_d.transpose((1, 0))
        label_p = label_p.transpose((1, 0))
        label_s = label_s.transpose((1, 0))

        pre_d, rec_d, fsc_d = evalue(x_d_c, label_d, label)
        print('Precision_d:', pre_d)
        print('Recall_d:', rec_d)
        print('F-score_d:', fsc_d)
        pre_ds.append(pre_d)
        rec_ds.append(rec_d)
        fsc_ds.append(fsc_d)

        pre_p, rec_p, fsc_p = evalue(x_p_c, label_p, label)
        print('Precision_p:', pre_p)
        print('Recall_p:', rec_p)
        print('F-score_p:', fsc_p)
        pre_ps.append(pre_p)
        rec_ps.append(rec_p)
        fsc_ps.append(fsc_p)

        pre_s, rec_s, fsc_s = evalue(x_s_c, label_s, label)
        print('Precision_s:', pre_s)
        print('Recall_s:', rec_s)
        print('F-score_s:', fsc_s)
        pre_ss.append(pre_s)
        rec_ss.append(rec_s)
        fsc_ss.append(fsc_s)

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

pre_d_avg = np.mean(pre_ds)
rec_d_avg = np.mean(rec_ds)
fsc_d_avg = np.mean(fsc_ds)

pre_p_avg = np.mean(pre_ps)
rec_p_avg = np.mean(rec_ps)
fsc_p_avg = np.mean(fsc_ps)

pre_s_avg = np.mean(pre_ss)
rec_s_avg = np.mean(rec_ss)
fsc_s_avg = np.mean(fsc_ss)

print('Precision_d_avg:', pre_d_avg)
print('Recall_d_avg:', rec_d_avg)
print('F-score_d_avg:', fsc_d_avg)

print('Precision_p_avg:', pre_p_avg)
print('Recall_p_avg:', rec_p_avg)
print('F-score_p_avg:', fsc_p_avg)

print('Precision_s_avg:', pre_s_avg)
print('Recall_s_avg:', rec_s_avg)
print('F-score_s_avg:', fsc_s_avg)

