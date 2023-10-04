from statistics import mode
import torch

def load_checkpoint_free(pretrained_dict, model, return_keys=False):
    removed_keys = []
    loaded_keys = []
    this_model_dict = model.state_dict()
    for k,v in pretrained_dict.items():
        if k in this_model_dict.keys():
            if this_model_dict[k].shape == v.shape:
                this_model_dict[k] = v
                loaded_keys.append(k)
            else:
                removed_keys.append(k)
        else:
            removed_keys.append(k)
    model.load_state_dict(this_model_dict)        
    if return_keys:
        return model, loaded_keys, removed_keys
    return model

def loadPretrain(model, pretrained_dir):
    pretrained_dict = torch.load(pretrained_dir)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['state_dict']
    removed_keys = []
    this_model_dict = model.state_dict()
    for k,v in pretrained_dict.items():
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        if k.startswith("module."):
            k = k[len("module."):]
            
        if k in this_model_dict.keys():
            if this_model_dict[k].shape == v.shape:
                this_model_dict[k] = v
            else:
                removed_keys.append(k)
        else:
            removed_keys.append(k)
    model.load_state_dict(this_model_dict)        
    return model, removed_keys