"""Budget arithmetic and temporary intra-op threads for CPU learner updates."""
from contextlib import contextmanager


def validate_learner_threads(threads):
    if type(threads) is not int or threads not in (1, 2, 4):
        raise ValueError("learner threads must be one of 1, 2, 4")
    return threads


def minimum_cpu_budget(workers, threads=1):
    validate_learner_threads(threads)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    # workers=1 uses map in the learner process: rollout and update are serial.
    # A process pool conservatively reserves both workers and learner threads.
    return threads if workers == 1 else workers+threads


@contextmanager
def learner_update_threads(threads=1):
    """Only update kernels may expand; every exit restores single-thread mode.

    Torch is deliberately imported lazily so launcher budget validation does not
    initialize a numerical runtime. The default path does not set threads or
    touch model, optimizer, Python RNG or Torch RNG state.
    """
    validate_learner_threads(threads)
    import torch
    try:
        if torch.get_num_interop_threads() != 1:
            raise ValueError("learner inter-op threads must remain one")
        if torch.get_num_threads() != 1:
            raise ValueError("learner update must enter from single-thread mode")
        if threads != 1:
            torch.set_num_threads(threads)
        yield
    finally:
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
