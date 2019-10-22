from v1_interpolation_27 import interpolation_between_two_action27
from v1_interpolation_32 import interpolation_between_two_action32
import os
import scipy.io as sio

if __name__ == '__main__':
    path = './data/h3.6m/Train/train_xyz/'
    files = os.listdir(path)
    files = [os.path.splitext(file)[0] for file in files]
    print("——————START——————")
    files.sort()
    # model = 'just_save'   #  save video+mat
    model = 'save_mat'  # only save mat
    for i in range(len(files)):
        for j in range(len(files)):
            if i < j:
                print(i, files[i], j, files[j])
                shape1, shape2 = interpolation_between_two_action32(files[i], files[j], model=model)
                if shape1 == shape2 and shape1 == 27:
                    shape1, shape2 = interpolation_between_two_action27(files[i], files[j], model=model)
                print(shape1, shape2)
                print("————————————")

