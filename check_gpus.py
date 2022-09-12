import torch

for i in range(torch.cuda.device_count()):
    print(f"{i} => {torch.cuda.get_device_name(i)}")