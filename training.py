import torch
import lpips
import torchvision
import os
import time
from model.model import *
from model.unet import *
from model.submodules import *
from dataset import EventData


# Inputs
root='/home/lab101m5/event/ecoco_depthmaps_test/train/'
event_dir = '/VoxelGrid-betweenframes-5'
frame_dir = '/frames'

# GPU
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Instantiate model as E2VIDRecurrent
conf = {'num_bins': 5, 'recurrent_block_type': 'convlstm', 'num_encoders': 3, 'num_residual_blocks': 2, 'use_upsample_conv': False, 'norm': 'BN'}
model = E2VIDRecurrent(conf)

# Model to GPU
model = model.to(device)

# Loss function (LPIPS VGG)
loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_vgg = loss_fn_vgg.to(device)

# Use ADAM optimizer with learning rate of 0.0001
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start = time.time()

# Training
for t in range(160):  # TRAIN FOR 160 EPOCHS
    print('EPOCH ', t)

    for seq in sorted(os.listdir(root)):
        print('Training ' + seq)

        # Create dataset for the sequence
        dataset = EventData(root=root, seq=seq, event_dir=event_dir, frame_dir=frame_dir)
        # Load sequence in batches of 2
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2) 

        prev_states=None

        for i, data in enumerate(dataloader, 0):
            x, y = data[0].to(device), data[1].to(device)

            # Forward pass: compute predicted y by passing x to the model.
            y_pred, states = model.forward(x, prev_states)

            # Compute and print loss.
            loss = loss_fn_vgg(y_pred, y)

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters.
            loss.sum().backward(retain_graph=True)

            # Calling the step function on an Optimizer makes an update to its
            # parameters.
            optimizer.step()

            with torch.no_grad():
                prev_states = states

# Total training time
end = time.time()
print('Training time: ', end-start, ' s')

# GPU tensor to CPU, save model
model = model.cpu()
#torch.save(model, PATH) #'/home/lab101m5/event/rpg_e2vid/trained_model.pth') 
