from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), fig_save_name=None):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    if fig_save_name is None:
        plt.imshow(image_grid.permute(1, 2, 0).squeeze().numpy())
    else:
        plt.imsave(fig_save_name, image_grid.permute(1, 2, 0).squeeze().numpy())
    plt.show()
    plt.close()


def get_noise(n_samples, z_dim, device='cpu'):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    :param n_samples: the number of samples to generate, a scalar
    :param z_dim: the dimension of the noise vector, a scalar
    :param device: the device type
    :return: the mini-batch noise tensor generated
    """
    return torch.randn(n_samples, z_dim, device=device)
