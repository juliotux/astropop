import numpy as np


def xy2r(x, y, data, xc, yc):
    r = np.sqrt((x-xc)**2 + (y-yc)**2)
    return np.ravel(r), np.ravel(data)


def trim_array(data, box_size, position, indices=None):
    x, y = position
    dx = dy = float(box_size)/2

    x_min = max(int(x-dx), 0)
    x_max = int(x+dx)+1
    y_min = max(int(y-dy), 0)
    y_max = int(y+dy)+1

    d = data[y_min:y_max, x_min:x_max]

    if indices is None:
        return d, x-x_min, y-y_min
    else:
        xi = indices[1][y_min:y_max, x_min:x_max]
        yi = indices[0][y_min:y_max, x_min:x_max]
        return d, xi, yi
