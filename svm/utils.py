import numpy as np

def dataset_split(data, label, test_size=.15, valid_size=.15, shuffle=True):
    label = label.reshape((-1,))
    num = len(label)
    data = data.reshape((num, -1))
    test_num = int(num * test_size)
    valid_num = int(num * valid_size)
    train_num = num - test_num - valid_num
    if shuffle:
        index = np.random.permutation(num)
    else:
        index = np.arange(num)

    train_data, train_label = data[index[0:train_num]], label[index[0:train_num]]
    valid_data, valid_label = data[index[train_num:train_num + valid_num]], label[
        index[train_num:train_num + valid_num]]
    test_data, test_label = data[index[train_num + valid_num:num]], label[index[train_num + valid_num:num]]
    return train_data, train_label, test_data, test_label, valid_data, valid_label