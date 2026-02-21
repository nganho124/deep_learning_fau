import torch as t
from trainer import Trainer
import sys
import torchvision as tv

epoch = int(sys.argv[1])
#TODO: Enter your model here
resnet = model.ResNet()



crit = t.nn.BCELoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))

print(f"✅ Exported checkpoint {epoch} to checkpoint_{epoch:03d}.onnx")
