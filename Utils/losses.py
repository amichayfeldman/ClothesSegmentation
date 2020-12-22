import torch


def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    """
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator
                  predictions and the true labels and returns a adversarial
                  loss (which you aim to minimize)
        recon_criterion: the reconstruction loss function; takes the generator
                    outputs and the real images and returns a reconstructuion
                    loss (which you aim to minimize)
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    """
    fake_img = gen(condition)
    disc_out = disc(x=fake_img, y=condition)
    gen_loss = adv_criterion(disc_out, torch.ones_like(disc_out)) + lambda_recon * recon_criterion(fake_img, real)
    return gen_loss
