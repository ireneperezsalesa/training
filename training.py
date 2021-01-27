import torch
import lpips
import torchvision
import os
import time
import math
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

        try:
            # Create dataset for the sequence
            dataset = EventData(root=root, seq=seq, event_dir=event_dir, frame_dir=frame_dir)
        except:
            print('Error creating dataset at epoch ', t, 'seq', seq)

        try: 
            # Load sequence in batches of 2
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False) 
        except:
            print('Error loading dataset at epoch ', t, 'seq', seq)

        # Initialize states
        states=None

        for i, data in enumerate(dataloader, 0):
            x, y = data[0].to(device), data[1].to(device)

            try:
                # Forward pass: compute predicted y by passing x to the model.
                y_pred, states = model.forward(x, states)
            except:
                print('Error in forward pass at epoch ', t, 'seq', seq)

            try:
                # Compute and print loss.
                loss = loss_fn_vgg(y_pred, y)
            except:
                print('Error computing loss at epoch ', t, 'seq', seq)

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

                # Save model, checkpoint several times per epoch
                if seq.endswith('49'):
                    print('Saving')
                    time_now = time.time()
                    train_time = time_now - start

                    cp_file_name = '/home/lab101m5/event/rpg_e2vid/checkpoints/cp_' + str(t) + seq + '.tar'
                    model_name = '/home/lab101m5/event/rpg_e2vid/checkpoints/m_' + str(t) + seq + '.pth'

                    torch.save({
                        'last_epoch': t,
                        'last_seq': seq,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'train_time': train_time
                    }, cp_file_name)
                    torch.save(model, model_name) 

# Total training time
end = time.time()
print('Training time: ', end-start, ' s')
