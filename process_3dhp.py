import numpy as np
from tqdm import tqdm
import os.path as osp

data = np.load('/home/SENSETIME/zengwang/mydata/mmhuman/mpi_inf_3dhp_train.npz', allow_pickle=True)
save_dict = dict(data)

imgs = save_dict['image_path']

imgs_new = []
for img in tqdm(imgs):
    tmp = img.split('/')
    sub = int(tmp[0].split('S')[1])
    seq = int(tmp[1].split('Seq')[1])
    vid = int(tmp[2].split('video_')[1])
    num = int(tmp[3].split('.jpg')[0])

    img_new = []
    img_new.append('S{}'.format(sub))
    img_new.append('Seq{}'.format(seq))
    img_new.append('images')
    img_new.append('S{}_Seq{}_V{}'.format(sub, seq, vid))
    img_new.append('img_S{}_Seq{}_V{}_{:0>6d}.jpg'.format(sub, seq, vid, num))
    img_new = osp.join(*img_new)
    imgs_new.append(img_new)

save_dict['image_path'] = imgs_new
np.savez('/home/SENSETIME/zengwang/mydata/mmhuman/mpi_inf_3dhp_train_mine.npz', **save_dict)
