import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.randn(10).to(device)
print(tensor.device)  # Should print "cuda:0" if on GPU