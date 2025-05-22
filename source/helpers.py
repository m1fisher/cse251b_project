
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

