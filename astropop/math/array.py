import numpy as np


def xy2r(x, y, data, xc, yc):
    r = np.hypot((x-xc), (y-yc))
    return np.ravel(r), np.ravel(data)


def trim_array(data, box_size, position, indices=None):
    x, y = position
    dx = dy = float(box_size)/2

    x_min = max(int(x-dx), 0)
    x_max = min(int(x+dx)+1, data.shape[1])
    y_min = max(int(y-dy), 0)
    y_max = min(int(y+dy)+1, data.shape[0])

    d = data[y_min:y_max, x_min:x_max]

    if indices is None:
        return d, x-x_min, y-y_min
    else:
        xi = indices[1][y_min:y_max, x_min:x_max]
        yi = indices[0][y_min:y_max, x_min:x_max]
        return d, xi, yi
