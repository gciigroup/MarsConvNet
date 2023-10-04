import numpy as np
import matplotlib.pyplot as plt
from options import *
from matplotlib.pyplot import MultipleLocator
import os


if not os.path.exists(opt.image_path):
    os.mkdir(opt.image_path)  

evals_path = './evals/evals.npy'   # evals

evals = np.load(evals_path, allow_pickle=True)
# print(evals)

dic = dict(evals.tolist())
# print(dic['pre_ds'])


loss_train_path = './loss/loss_total.npy'      # loss train

loss_train = np.load(loss_train_path, allow_pickle=True)

# print(loss_train)


loss_valid_path = './loss/loss_total_test.npy'      # loss test

loss_valid = np.load(loss_valid_path, allow_pickle=True)

a = list(range(1, 21))
# print(loss_valid)

ax=plt.gca()
plt.plot(a, loss_train, label='loss_train')
plt.legend(loc='best')
plt.ylabel('Loss Value')
plt.xlabel('Training Epochs')
x_major_locator=MultipleLocator(5)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig(opt.image_path + 'loss_train.png', dpi=300)
plt.close()

ax=plt.gca()
plt.plot(a, loss_valid, label='loss_valid')
plt.legend()
plt.ylabel('Loss Value')
plt.xlabel('Training Epochs')
x_major_locator=MultipleLocator(5)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig(opt.image_path + 'loss_valid.png', dpi=300)
plt.close()

ax=plt.gca()
plt.plot(a, loss_train, 'v-', label='loss_train')   # 三角
plt.plot(a, loss_valid, 's-', label='loss_valid')   # 方块
plt.legend(loc='best')
plt.ylabel('Loss Value')
plt.xlabel('Training Epochs')
x_major_locator=MultipleLocator(5)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig(opt.image_path + 'loss.png', dpi=300)
plt.close()

ax=plt.gca()

a1 = dic['pre_ds']
a2 = dic['rec_ds']
a3 = dic['fsc_ds']

# a1[-1] = dic2['pre_ds'][-1]
# a2[-1] = dic2['rec_ds'][-1]
# a3[-1] = dic2['fsc_ds'][-1]

plt.plot(a, a1, 'o-', label='pre_d')   # 圆点
plt.plot(a, a2, 'v-', label='rec_d')   # 三角
plt.plot(a, a3, 's-', label='fsc_d')   # 方块
plt.legend(loc='best')
plt.ylabel('Evaluation Index')
plt.xlabel('Training Epochs')
x_major_locator=MultipleLocator(5)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig(opt.image_path + 'eval_d.png', dpi=300)
plt.close()

ax=plt.gca()

b1 = dic['pre_ps']
b2 = dic['rec_ps']
b3 = dic['fsc_ps']

# b1[-1] = dic2['pre_ps'][-1]
# b2[-1] = dic2['rec_ps'][-1]
# b3[-1] = dic2['fsc_ps'][-1]

plt.plot(a, b1, 'o-', label='pre_p')   # 圆点
plt.plot(a, b2, 'v-', label='rec_p')   # 三角
plt.plot(a, b3, 's-', label='fsc_p')   # 方块
plt.legend(loc='best')
plt.ylabel('Evaluation Index')
plt.xlabel('Training Epochs')
x_major_locator=MultipleLocator(5)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig(opt.image_path + 'eval_p.png', dpi=300)
plt.close()

ax=plt.gca()

c1 = dic['pre_ss']
c2 = dic['rec_ss']
c3 = dic['fsc_ss']

# c1[-1] = dic2['pre_ss'][-1]
# c2[-1] = dic2['rec_ss'][-1]
# c3[-1] = dic2['fsc_ss'][-1]

plt.plot(a, c1, 'o-', label='pre_s')   # 圆点
plt.plot(a, c2, 'v-', label='rec_s')   # 三角
plt.plot(a, c3, 's-', label='fsc_s')   # 方块
plt.legend(loc='lower right')
plt.ylabel('Evaluation Index')
plt.xlabel('Training Epochs')
# plt.grid()
x_major_locator=MultipleLocator(5)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig(opt.image_path + 'eval_s.png', dpi=300)


print('plot finish!')

