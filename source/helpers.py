
def path_length_ratio(data):
    """
    GPT4o generated. Makes some assumptions about input dimensionality.

    data: np.ndarray
    """
    deltas   = np.diff(data, axis=2)                   # shape (10000,50,109,2)
    step_d   = np.linalg.norm(deltas, axis=-1)        # (10000,50,109)
    L        = step_d.sum(axis=-1)                    # (10000,50)
    start    = data[..., 0, :]                         # (10000,50,2)
    end      = data[..., -1, :]                        # (10000,50,2)
    D        = np.linalg.norm(end - start, axis=-1)   # (10000,50)

    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        nonlin = L / D

    # set zero-distance sequences to zero
    nonlin = np.where(D==0, 0.0, nonlin)

    return nonlin

def min_distance(data):
    """
    Return the min distance of any vehicle to the ego vehicle in each scene.
    """
    # assume `data` is your array of shape (10000, 50, x, 6)
    # extract only the x,y coordinates:
    pos = data[..., :2]                 # shape: (10000, 50, x, 2)

    # separate ego and others:
    ego   = pos[:, 0:1, :, :]          # shape: (10000,  1, x, 2)
    other = pos[:, 1:,    :, :]        # shape: (10000, 49, x, 2)

    # compute vector difference (broadcast ego across the “agent” axis):
    diff = other - ego                 # shape: (10000, 49, x, 2)

    # squared distances, summed over x/y:
    sq_dist = np.sum(diff**2, axis=-1) # shape: (10000, 49, x)

    # for each scene, find the minimum squared‐distance over all agents & timesteps:
    min_sq = np.min(sq_dist, axis=(1,2))  # shape: (10000,)

    # finally, take the sqrt to get Euclidean distance:
    min_dist_test = np.sqrt(min_sq)             # shape: (10000,)
    return min_dist_test
