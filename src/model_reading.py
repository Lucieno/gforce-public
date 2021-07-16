import functools
import re


def get_net_config_name(model_name_base):
    # if model_name_base == "minionn_maxpool":
    #     net_state_name = "./model/minionn_maxpool_swalp_893_net.pth"
    #     config_name = "./model/minionn_maxpool_swalp_893_exp_configs.npy"
    # else:
    loss_name_base = f"./model/{model_name_base}"
    config_name = loss_name_base + "_swalp_exp_configs.npy"
    net_state_name = loss_name_base + "_swalp_net.pth"
    return net_state_name, config_name

def get_hooking_lst(model_name_base):
    if model_name_base in ["vgg16_cifar100_swalp", "vgg16_cifar10_swalp", "vgg16_cifar10", "vgg16_cifar100"]:
        lst = [("input_layer", "input_layer")]
        for i in range(1, 14):
            lst.append(("trun%d"%i, "conv%d"%i))
        lst += [("trun_fc_1", "fc1"), ("trun_fc_2", "fc2"), ("fc3", "fc3")]
        return lst
    elif model_name_base in ["minionn_maxpool"]:
        lst = [("input_layer", "input_layer")]
        for i in range(1, 8):
            lst.append(("trun%d" % i, "conv%d" % i))
        lst += [("fc1", "fc1")]
        return lst
    else:
        raise Exception(f"Unknown for hooking_lst: {model_name_base}")

def get_num_layer(layer_prefix, store_configs):
    # all_key_str = ','.join(store_configs.keys())
    lst = store_configs.keys()
    return len(list(filter(
        lambda x: re.match(layer_prefix + "\d+ForwardX", x) is not None,
        lst)))
    # return fast_get_num_layer(layer_prefix, all_key_str)

@functools.lru_cache(maxsize=100, typed=True)
def fast_get_num_layer(layer_prefix, all_key_str):
    return len(re.findall(layer_prefix+"\d+ForwardX", all_key_str))

def get_last_conv_name(store_configs):
    return "conv%d"%get_num_layer("conv", store_configs)

def get_last_fc_name(store_configs):
    return "fc%d"%get_num_layer("fc", store_configs)
