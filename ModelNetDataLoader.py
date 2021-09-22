# Ayman: added  rotate_point_cloud_by_angle
import numpy as np
import warnings
import h5py
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
from sklearn.neighbors import KDTree

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    # normal = f['normal'][:]
    # data = np.concatenate([data,normal],axis=2)
    seg = []
    return (data, label, seg)

def load_data(dir,classification = False):
    data_train0, label_train0,Seglabel_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1,Seglabel_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2,Seglabel_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3,Seglabel_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4,Seglabel_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_test0, label_test0,Seglabel_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1, label_test1,Seglabel_test1 = load_h5(dir + 'ply_data_test1.h5')

    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel

class ModelNetDataLoader(Dataset):
    def __init__(self, data, labels, rotation = True):
        self.data = data
        self.labels = labels
        self.rotation = rotation

    def __len__(self):
        return len(self.data)

    

    def rotate_point_cloud_by_angle(self, data, rotation_angle):
        """
        Rotate the point cloud along up direction with certain angle.
        :param batch_data: Nx3 array, original batch of point clouds
        :param rotation_angle: range of rotation
        :return:  Nx3 array, rotated batch of point clouds
        """
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]],dtype='float32')
        rotated_data = np.dot(data[...,0:3], rotation_matrix)
        # rotated_datan = self.findPointNormals(data)
        # rotated_datan = np.float32(rotated_datan)
       # # rotated_datan = np.dot(data[..., 3:6], rotation_matrix)
        # rotated_data = np.concatenate([rotated_data,rotated_datan],axis=1)
        return rotated_data



    def __getitem__(self, index):
        if self.rotation:
            pointcloud = self.data[index]
            if self.rotation:
                angle = np.random.uniform() * 2 * np.pi  # np.random.randint(self.rotation[0], self.rotation[1]) * np.pi / 180
                pointcloud = self.rotate_point_cloud_by_angle(pointcloud, angle)
            return pointcloud, self.labels[index]
        else:
            return self.data[index], self.labels[index]



