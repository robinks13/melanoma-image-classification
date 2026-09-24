import torch

print("Versionde PyTorch :", torch.__version__)
print("CUDA disponible ?", torch.cuda.is_available())
print("Nom duGPU :", torch.cuda.get_device_name(0))

# Creer unpetit tenseur sur GPUpour tester
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
print("Tenseursur GPU :", x)
print("Devicedu tenseur :", x.device)