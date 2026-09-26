"""
Visual testing utilities for the trained VAE.

Run after training (expects weights/vae_model.pt to exist):
    python scripts/visualize_vae_output.py

A file picker will open so you can choose the image(s) to test with.
"""

from pathlib import Path

import tkinter as tk
from tkinter import filedialog

import torch
import matplotlib.pyplot as plt

from satisfactory_vision.train_vae import VAE, load_image

DIM = (64, 64)
CHANNELS = 3
LATENT_DIM = 2  # must match training
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "weights" / "vae_model.pt"

IMAGE_FILETYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("All files", "*.*"),
]


def load_model(checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    vae = VAE(latent_dim=LATENT_DIM, input_dim=DIM[0] * DIM[1] * CHANNELS, hidden_dim=512).to(device)
    vae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    vae.eval()
    return vae


def tensor_to_image(flat_tensor):
    """Reshape a flat decoder output [C*H*W] back to [H, W, C] numpy for plotting."""
    img = flat_tensor.detach().cpu().reshape(CHANNELS, DIM[1], DIM[0])
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    return img


def pick_images(title="Select image(s)", multiple=False):
    """Opens a native file picker and returns the selected path(s)."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # bring the dialog to the front

    if multiple:
        paths = filedialog.askopenfilenames(title=title, filetypes=IMAGE_FILETYPES)
    else:
        paths = filedialog.askopenfilename(title=title, filetypes=IMAGE_FILETYPES)

    root.destroy()

    if not paths:
        raise RuntimeError("No file selected.")
    return paths


@torch.no_grad()
def show_reconstructions(vae, device):
    """Pick one or more images, show original vs. reconstructed side by side."""
    paths = pick_images(title="Select image(s) to reconstruct", multiple=True)
    originals = torch.stack([load_image(p, DIM) for p in paths]).to(device)

    reconstructed, mu, log_var = vae(originals)
    n = len(paths)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(originals[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(tensor_to_image(reconstructed[i]))
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstructed", fontsize=10)
    fig.suptitle("Reconstruction quality")
    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=150)
    plt.show()


@torch.no_grad()
def show_latent_grid(vae, device, grid_size=12, span=3.0):
    """
    Only meaningful because latent_dim == 2.
    Walks a grid over z1, z2 in [-span, span] and decodes each point.
    Reveals whether the latent space is smooth/continuous.
    """
    assert vae.latent_dim == 2, "Latent grid visualization only works for latent_dim=2"

    axis_vals = torch.linspace(-span, span, grid_size)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))

    for i, z1 in enumerate(axis_vals):
        for j, z2 in enumerate(axis_vals):
            z = torch.tensor([[z1, z2]], dtype=torch.float32, device=device)
            decoded = vae.decode(z)[0]
            axes[grid_size - 1 - i, j].imshow(tensor_to_image(decoded))
            axes[grid_size - 1 - i, j].axis("off")

    fig.suptitle("Latent space grid (z1 x z2)")
    plt.tight_layout()
    plt.savefig("latent_grid.png", dpi=150)
    plt.show()


@torch.no_grad()
def show_interpolation(vae, device, steps=10):
    """Pick two images, interpolate linearly between their mu vectors, decode each step."""
    path_a = pick_images(title="Select the FIRST image for interpolation")
    path_b = pick_images(title="Select the SECOND image for interpolation")

    img_a = load_image(path_a, DIM).unsqueeze(0).to(device)
    img_b = load_image(path_b, DIM).unsqueeze(0).to(device)

    mu_a, _ = vae.encode(img_a)
    mu_b, _ = vae.encode(img_b)

    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2))
    for i, alpha in enumerate(torch.linspace(0, 1, steps)):
        z = mu_a * (1 - alpha) + mu_b * alpha
        decoded = vae.decode(z)[0]
        axes[i].imshow(tensor_to_image(decoded))
        axes[i].axis("off")

    fig.suptitle("Latent space interpolation")
    plt.tight_layout()
    plt.savefig("interpolation.png", dpi=150)
    plt.show()


@torch.no_grad()
def show_random_samples(vae, device, n=8, std=1.0):
    """Sample z ~ N(0, std^2 I) and decode — tests whether the prior actually generates plausible images."""
    z = torch.randn(n, vae.latent_dim, device=device) * std
    decoded = vae.decode(z)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    for i in range(n):
        axes[i].imshow(tensor_to_image(decoded[i]))
        axes[i].axis("off")

    fig.suptitle("Random samples from prior")
    plt.tight_layout()
    plt.savefig("random_samples.png", dpi=150)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_model(CHECKPOINT_PATH, device)

    show_reconstructions(vae, device)
    show_latent_grid(vae, device, grid_size=12, span=3.0)
    show_interpolation(vae, device, steps=10)
    show_random_samples(vae, device, n=8)


if __name__ == "__main__":
    main()