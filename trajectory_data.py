import numpy as np

data_path = 'trajectory_data/'
suffix = '.npz'
def save_arrays(xs, us, filename='data_compressed'):
    """
    Saves lists of numpy arrays `xs` and `us` into a compressed npz file.

    Parameters:
        xs (list of np.ndarray): List of state arrays.
        us (list of np.ndarray): List of control arrays.
        filename (str): Filename to save the compressed arrays.
    """
    full_path = data_path + filename + suffix
    np.savez_compressed(full_path, xs=np.array(xs), us=np.array(us))

def load_arrays(filename='data_compressed'):
    """
    Loads lists of numpy arrays `xs` and `us` from a compressed npz file.

    Parameters:
        filename (str): Filename to load the arrays from.

    Returns:
        tuple: A tuple containing two lists of np.ndarray, `xs` and `us`.
    """
    full_path = data_path + filename + suffix
    data = np.load(full_path, allow_pickle=True)
    xs = [np.array(x) for x in data['xs']]
    us = [np.array(u) for u in data['us']]
    return xs, us