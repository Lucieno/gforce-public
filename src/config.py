import torch

class Config(object):
    world_size = 2
    both_rank = -1
    server_rank = 0
    client_rank = 1
    # n_23, q_23 = 8192, 7340033
    # n_16, q_16 = 2048, 12289
    n_23, q_23 = 16384, 7340033
    n_16, q_16 = 16384, 65537
    safe_degree = 16384
    noise_bit = 320
    global_random_seed = None # None -> int(time()), o.w. to be the seed
    trunc_seed = None # None -> int(time()), o.w. to be the seed
    is_shuffle = True # affect security but not correctness. can be tuned off when testing correctness
    use_cuda = True
    device_name = "cuda:0" if use_cuda else "cpu"
    device = torch.device(device_name)
