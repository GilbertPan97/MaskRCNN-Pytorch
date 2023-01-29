import os
import torch
import torchvision

model = torchvision.models.resnet18()

example = torch.rand(1, 3, 224, 224)

trace_script_module = torch.jit.script(model, example)

trace_script_module.save("test_model.pt")

