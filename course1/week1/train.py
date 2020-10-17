import os
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST  # Training dataset
from course1.week1.models import Generator, Discriminator
from course1.week1.utils import *

torch.manual_seed(2020)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """
    Return the loss of the discriminator given inputs.

    :param gen: the generator model, which returns an image given z-dimensional noise
    :param disc: the discriminator model, which returns a single-dimensional prediction of real/fake
    :param criterion: the loss function, which should be used to compare
           the discriminator's predictions to the ground truth reality of the images
           (e.g. fake = 0, real = 1)
    :param real: a batch of real images
    :param num_images: the number of images the generator should produce,
           which is also the length of the real images
    :param z_dim: the dimension of the noise vector, a scalar
    :param device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    """
    noise = get_noise(n_samples=num_images, z_dim=z_dim, device=device)
    fake_img = gen(noise).detach()  # DO NOT FORGET to detach

    fake_logits = disc(fake_img)
    loss_fake = criterion(fake_logits, torch.zeros_like(fake_logits))

    real_logits = disc(real)
    loss_real = criterion(real_logits, torch.ones_like(real_logits))

    return torch.mean(loss_fake + loss_real)


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    """


    :param gen: the generator model, which returns an image given z-dimensional noise
    :param disc: the discriminator model, which returns a single-dimensional prediction of real/fake
    :param criterion:the loss function, which should be used to compare
           the discriminator's predictions to the ground truth reality of the images
           (e.g. fake = 0, real = 1)
    :param num_images: the number of images the generator should produce,
           which is also the length of the real images
    :param z_dim: the dimension of the noise vector, a scalar
    :param device: the device type
    :return gen_loss: a torch scalar loss value for the current batch
    """
    noise = get_noise(n_samples=num_images, z_dim=z_dim, device=device)
    fake_img = gen(noise)

    fake_logits = disc(fake_img)
    gen_loss = criterion(fake_logits, torch.ones_like(fake_logits))

    return gen_loss


# Set your parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 2000
batch_size = 128
lr = 1e-5
device = 'cuda:0'
# Load MNIST dataset as tensors
dataset_root = "/home/liuzhian/hdd/datasets/MNIST"

dataloader = DataLoader(
    MNIST(root=dataset_root, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True  # Whether the generator should be tested
gen_loss = False
for epoch in range(n_epochs):
    print("Training epoch %d/%d:" % (epoch, n_epochs))
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()
        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        # For testing purposes, to keep track of the generator weights
        if test_generator:
            old_generator_weights = gen.gen._modules["0"].block._modules["0"].weight.detach().clone()

        ### Update generator ###
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()

        # For testing purposes, to check that your code changes the generator weights
        if test_generator:
            assert torch.any(gen.gen._modules["0"].block._modules["0"].weight.detach().clone() != old_generator_weights)

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if not os.path.exists("vis"):
            os.mkdir("vis")
        if cur_step % display_step == 0 and cur_step > 0:
            print(
                f"Step %d: Generator loss: %.4f, discriminator loss: %.4f"
                % (cur_step, mean_generator_loss, mean_discriminator_loss))
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake, fig_save_name="./vis/%d_fake.png" % cur_step)
            show_tensor_images(real, fig_save_name="./vis/%d_real.png" % cur_step)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
