import numpy as np
from tqdm import tqdm
h36m = np.load('/home/SENSETIME/zengwang/mydata/mmhuman/h36m_train.npz', allow_pickle=True)
h36m_mine = np.load('/home/SENSETIME/zengwang/mydata/npz_files/h36m_train_new.npz', allow_pickle=True)
save_dict = dict(h36m)

imgs = h36m['image_path']
imgs_mine = h36m_mine['imgname']
pose = h36m_mine['pose']
shape = h36m_mine['shape']

flag = True
for i in tqdm(range(len(imgs))):
    img = imgs[i]
    img_m = imgs_mine[i]
    if img.replace('/images/', '/') != img_m:
        flag = False
        print(img)
        print(img_m)

if flag:
    betas = shape
    pose = pose.reshape(-1, 24, 3)
    global_orient = pose[:, 0, :]
    body_pose = pose[:, 1:, :]
    smpl = dict(betas=betas, global_orient=global_orient, body_pose=body_pose)
    save_dict['smpl'] = smpl
    save_dict['image_path'] = imgs_mine
    np.savez('/home/SENSETIME/zengwang/mydata/mmhuman/h36m_train_mine.npz', **save_dict)
