# Your First GAN

In this notebook, you're going to create your first generative adversarial network (GAN) for this course! Specifically, 
you will build and train a GAN that can generate hand-written images of digits (0-9).

Don't expect anything spectacular: this is only the first lesson. The results will get better with later lessons 
as you learn methods to help keep your generator and discriminator at similar levels. 

# Configuration
You can modify the [parameters-setting](./train.py#L65-#L74) before training.

# Train
```shell
$ python train.py
```
The visualized images during training process will be saved at `./vis` dir by default.
