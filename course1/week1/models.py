import torch
import torch.nn as nn
from torchvision.datasets import MNIST


class BlockGen(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Constructor for creating a block of the generator's neural network
        given input and output dimensions.
        :param in_dim: the dimension of the input vector, a scalar
        :param out_dim: the dimension of the output vector, a scalar
        """
        super(BlockGen, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, z_dim, im_dim=28 * 28, hidden_dim=128):
        """
        Constructor for creating a generator of GAN
        :param z_dim: the dimension of the noise vector, a scalar
        :param im_dim:the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        :param hidden_dim:the inner dimension, a scalar
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.hidden_dim = hidden_dim

        # build the net
        self.gen = nn.Sequential(
            BlockGen(in_dim=z_dim, out_dim=hidden_dim),
            BlockGen(in_dim=hidden_dim, out_dim=2 * hidden_dim),
            BlockGen(in_dim=2 * hidden_dim, out_dim=4 * hidden_dim),
            BlockGen(in_dim=4 * hidden_dim, out_dim=8 * hidden_dim),
            nn.Linear(8 * hidden_dim, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.

        :param noise: a noise tensor with dimensions (n_samples, z_dim)
        :return: the n images generated
        """
        return self.gen(noise)


class BlockDisc(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Constructor for creating a block of the discriminator's neural network
        given input and output dimensions.

        Here, we design the block to be a combination of linear transformation
        followed by a leaky ReLu activation with negative slope of 0.2.

        :param in_dim: the dimension of the input vector, a scalar
        :param out_dim: the dimension of the output vector, a scalar
        """
        super(BlockDisc, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.block = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, im_dim=28 * 28, hidden_dim=128):
        """
        Constructor for creating a discriminator of GAN
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
        """
        super(Discriminator, self).__init__()
        self.im_dim = im_dim
        self.hidden_dim = hidden_dim
        self.disc = nn.Sequential(
            BlockDisc(in_dim=im_dim, out_dim=hidden_dim * 4),
            BlockDisc(in_dim=hidden_dim * 4, out_dim=hidden_dim * 2),
            BlockDisc(in_dim=hidden_dim * 2, out_dim=hidden_dim),
            nn.Linear(hidden_dim, 1),
            # not use a activation map here, since it will be included in a loss function
        )

    def forward(self, im):
        """
        Function for completing a forward pass of the discriminator: Given a noise tensor,
        returns a 1-dimension tensor representing fake/real.
        :param im: a flattened image tensor with dimension (im_dim)
        :return: the 1-dimension tensor representing fake/real
        """
        return self.disc(im)
