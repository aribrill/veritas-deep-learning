# process_data.py

import numpy as np

from veritas_header import pixels

def load_data(n_train, data_file, labels_file):
    raw_data = np.genfromtxt(data_file, delimiter=' ', max_rows=n_train)
    #test_data = np.genfromtxt('59521_data.txt', delimiter=' ', skip_header=n_train, max_rows=n_test)
    gammas = np.loadtxt(labels_file, dtype=int, delimiter=' ')
    labels = np.array([1 if i in gammas else 0 for i, _ in enumerate(
    xrange(len(raw_data)), start=1)])
    
    data = np.zeros((n_train, 4, 64, 64))
    for tel in xrange(4):
        start = tel*499
        end = (tel+1)*499
        for pixel, value in enumerate(raw_data[:, start:end].T, start=1):
            # Calculate grid coordinate of upper left hand corner of each pixel
            x_coord = 5 + 2*pixels[pixel][1] + pixels[pixel][0]
            y_coord = 5 + -2*pixels[pixel][0] + 26
            data[:, tel, x_coord, y_coord] = value
            data[:, tel, x_coord + 1, y_coord] = value
            data[:, tel, x_coord, y_coord + 1] = value
            data[:, tel, x_coord + 1, y_coord + 1] = value
   
    return data, labels


