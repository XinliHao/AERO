import os
import numpy as np

def normalize(a, min_a = None, max_a = None):
    if min_a is None:
        min_a = np.min(a, axis = 0)
        max_a = np.max(a, axis = 0)
    return (a - min_a) / (max_a - min_a)

def load_and_save(category, filename):
    temp = np.loadtxt(os.path.join(dataset_folder, filename), dtype=np.float64, delimiter=',')
    # 对mag进行归一化
    temp[:,1:] = normalize(temp[:,1:])
    np.save(os.path.join(output_folder, f"{dataset}_{category}.npy"), temp)
    print(category, temp.shape)
    return temp.shape

# 因为后续可以在不同的维度上进行计算，所以加载的标签是针对于多个维度的。
def load_and_save_label(category, filename, shape):
    # 传入的shape是包含着时间维的
    temp = np.zeros((shape[0],shape[1]-1))
    with open(os.path.join(dataset_folder, filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
        temp[start-1:end-1, indx] = 1
    print(category, temp.shape)
    np.save(os.path.join(output_folder, f"{dataset}_{category}.npy"), temp)

def load_data():
    file_list = os.listdir(dataset_folder)
    train_file = f'{dataset_folder}{dataset}_train.txt'
    test_file = f'{dataset_folder}{dataset}_test.txt'
    label_file = f'{dataset_folder}{dataset}_interpretation_label.txt'
    load_and_save('train', train_file)
    shape = load_and_save('test', test_file)
    load_and_save_label('labels', label_file, shape)



if __name__ == '__main__':
    dataset = '032_22790425-G0014'    #044_G0014-02820255，024_G0014-03250425，041_14110425-G0014
    dataset_folder = f'/home/wamdm/xinli/AstroSet-20220907/Dataset_txt/{dataset}/'
    output_folder = f"processed/{dataset}/"
    os.makedirs(output_folder, exist_ok=True)
    load_data()