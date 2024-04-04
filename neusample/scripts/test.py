
import torch 



import commentjson as json
import tinycudann as tcnn
import torch



def oneblob_encoding(tensor):
    with open("data/config_oneblob.json") as f:
        config = json.load(f)
    encoding = tcnn.Encoding(tensor.shape[1], config["encoding_oneblob"])
    return encoding(tensor)

def sh_encoding(tensor):
    with open("data/config_oneblob.json") as f:
        config = json.load(f)
    encoding = tcnn.Encoding(tensor.shape[1], config["encoding_sh"])
    return encoding(tensor)


wo = torch.rand(5,2)
encoded_wo = sh_encoding(wo)
print(wo.shape)
