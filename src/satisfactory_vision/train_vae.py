import torch
import torch.nn as nn
import torch.nn.functional as functional
from pathlib import Path

def load_image(image_path) -> torch.Tensor:
    """
    Loads an image from the given path and converts it to a PyTorch tensor.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize the image to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Compress to 128x128 pixels
    image = cv2.resize(image, (128, 128))

    # Convert to PyTorch tensor and add batch dimension
    tensor = torch.from_numpy(image).permute(2, 0, 1)

    return tensor
def get_image_paths(directory: str) -> list[Path]:
    """
    Returns a list of image paths in the given directory.
    """
    folder = Path(directory)

    file_array = [str(file) for file in folder.rglob("*") if file.is_file()]

    return file_array
class VAE(nn.Module):
    def __init__(self, latent_dim=2): # Two dimensional latent space. output will be a 2d tensor
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.Encoder(latent_dim)
        self.decoder = self.Decoder(latent_dim)

    def encode(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor into a mean and a log variance.
        """
        pass
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent variable z back into the original space.
        """
        pass
    def loss_calcualator(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the VAE.
        """
        pass
    def forward(self, x):
        pass


def main():
    # Get all images paths
    image_paths = get_image_paths("data")
    print(image_paths)

if __name__ == "__main__":
    main()