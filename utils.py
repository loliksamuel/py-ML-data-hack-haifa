import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
# from config import *


def filename_to_date(x):

    t = "/usr"#[os.path.splitext(i)[0][-8:] for i in x]
    assert (datetime(year=int(t[0][:4]), month=int(t[0][4:6]), day=int(t[0][6:8])))

    return t


def sort_files_by_date(file_names):

    t = filename_to_date(file_names)
    sorted_t_ind = np.argsort(t)
    file_names = [file_names[i] for i in sorted_t_ind]
    t = [t[i] for i in sorted_t_ind]

    return file_names, t


def train_test_split(file_names, train_ratio=0.8, past_future=True, test_year='201608'):

    if past_future:
        file_names, t = sort_files_by_date(file_names)
        if test_year is not None:
            print('split train test by year')
            try:
                train_ratio = np.where([test_year in i for i in t])[0][0]/len(t)
            except IndexError:
                print('train ratio by year failed')
    else:
        np.random.shuffle(file_names)
    n_train = int(train_ratio * len(file_names))
    train = file_names[:n_train]
    test = file_names[n_train:]

    return train, test


def get_files_limited(file_names, debug_limit):

    if debug_limit:
        shuffled_file_names = file_names[:]
        np.random.shuffle(shuffled_file_names)
        file_names_limited = shuffled_file_names[:np.minimum(debug_limit, len(file_names))]
    else:
        file_names_limited = file_names
    return file_names_limited


def list_partition(list_in, n_part=80, shuffle=False):

    if n_part > len(list_in):
        return [list_in]

    processed_list = list_in[:]
    if shuffle:
        np.random.shuffle(processed_list)

    part_list = []
    n_per_part, rem = divmod(len(processed_list), n_part)
    for part_ind in range(n_part):
        curr_part = []
        for in_part_ind in range(n_per_part):
            curr_part.append(processed_list[part_ind * n_per_part + in_part_ind])
        part_list.append(curr_part)

    if rem:
        part_list[-1] += processed_list[-rem:]

    return part_list


def get_fit_in_partition(load_handle, file_names, limit=2e8, shuffle=False):

    fit_in_n_part = len(file_names)

    part_opt = np.array([100, 30, 10, 3, 1])
    part_opt = part_opt[part_opt < fit_in_n_part]

    for curr_n_part in part_opt:
        files_partitioned = list_partition(file_names, n_part=curr_n_part)
        X = load_handle(files_partitioned[0])[0]
        if not len(X):
            fit_in_n_part = part_opt[-1]
            break
        if X.nbytes > limit:
            break
        else:
            fit_in_n_part = curr_n_part

    files_partitioned = list_partition(file_names, n_part=fit_in_n_part, shuffle=shuffle)

    return files_partitioned


def cyclic_gaussian_conv(x, period, mu=0, sigma=1):

    y = np.zeros((len(x), ))
    for k in [-1, 0, 1]:
        y += np.exp(-((k * period + x - mu)/sigma)**2)
    return y


def parse_digits(s, return_digits=True):

    if return_digits:
        return ''.join([j for j in s if j.isdigit()])
    else:
        return ''.join([j for j in s if not j.isdigit()])


def subplot_matrix(mat, titles, x=None):

    n = mat.shape[1]
    m = int(np.ceil(n / 2))
    f, ax_arr = plt.subplots(2, m)

    x_tick = np.arange(mat.shape[0])

    palette = ['b', 'r']

    for k in range(2):
        for j in range(m):
            ind = k * m + j
            if ind == n:
                break
            for l in range(mat.shape[2]):
                ax_arr[k, j].scatter(x_tick, mat[:, ind, l], s=50, c=palette[l])
            if x is not None:
                ax_arr[k, j].set_xticklabels(x)
            ax_arr[k, j].set_title(titles[ind])


def safe_del(d, k):

    if k in d.keys():
        del d[k]
    else:
        pass

    return d


def print_completed(counter, total, part=10, msg='completed'):

    counter += 1
    if counter % np.ceil(total / part) == 0:
        print(str(int(100 * counter / total)) + '% ' + msg)

    return counter


def dict_to_str(d):

    return '_'.join(['_'.join([k,  str(v)]) for k, v in d.items()])


def interpolate(x, y, xi):

    yi = []
    for i in range(x.shape[0]):
        curr_x = x[i, :]
        if not np.all(np.isfinite(curr_x)):
            yi.append(np.nan)
            continue
        ind_gt = np.where(curr_x >= xi)[0][0]
        ind_lt = np.where(curr_x < xi)[0][-1]
        dx = (xi - curr_x[ind_lt]) / (curr_x[ind_gt] - curr_x[ind_lt])
        yi.append(y[i, ind_lt] + dx * (y[i, ind_gt] - y[i, ind_lt]))

    return np.array(yi).reshape(-1, 1)


if __name__ == '__main__':

    pass
