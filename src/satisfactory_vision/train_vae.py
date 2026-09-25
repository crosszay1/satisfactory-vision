import torch
import torch.nn as nn
import torch.nn.functional as functional
from pathlib import Path
import cv2
import numpy as np

def load_image(image_path) -> torch.Tensor:
    """
    Loads an image from the given path and converts it to a PyTorch tensor.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize the image to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Compress to 128x128 pixels
    image = cv2.resize(image, (128, 128))

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(image).permute(2, 0, 1)

    return tensor
def get_image_paths(directory: str) -> list[str]:
    """
    Returns a list of image paths in the given directory.
    """
    folder = Path(directory)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"} # Have to do this because we have lebels.json which will mess stuff up if we try and load it
    file_array = [
        str(file)
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in image_extensions
    ]

    return file_array
def reLu(x): # Won't be used, just for understanding
    """
    Applies the ReLU activation function to the input tensor.
    Input: x - A PyTorch tensor
    Output: 0 if x < 0, else returns back x
    """
    output = max(0, x)
    return output
class VAE(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int, hidden_dim: int):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor into a mean and a log variance.
        """
        # Flatten to [batch, input_dim]
        tensor = tensor.reshape(-1, self.fc1.in_features)
        h1 = functional.relu(self.fc1(tensor)) # Hidden layer 1 activation = Relu applied to the output of the first hidden layer with tensor as the input

        mu = self.fc2_mu(h1) # Mean of the latent space
        log_var = self.fc2_log_var(h1) # Log variance of the latent space
        
        return mu, log_var

    def reparameterize(self, mu, log_var) -> torch.Tensor: # Non-deterministic
        """"
        Reparameterization turns mean and log variance into a latent variable z we can then feed to the decoder

        Explanation (Because I'm pretty much commenting all my code for once): (Need rewriting)
        # Output of the encoder is non-deterministic, meaning that the same input can produce different outputs. 
        # This bad!
        # So we rep
        """
        standard_deviation = torch.exp(0.5 * log_var) # Get standard deviation from log variance
        epsillon = torch.randn_like(standard_deviation) # Random noise with the same shape as standard deviation
        z = mu + epsillon * standard_deviation # Reparameterization trick: z = mu + sigma * epsilon
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
            """
            Decodes the latent variable z back into the original space.
            """
            h3 = functional.relu(self.fc3(z)) # Pass z through hidden layer
            h4 = functional.sigmoid(self.fc4(h3)) # Pass the output of the hidden layer through the output layer, and run sigmoid to get output in range [0,1]
            return h4

    def loss_calcualator(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the VAE.
        x: Original input tensor
        x_reconstructed: Reconstructed tensor from the decoder
        Outputs: Mean squared error loss as a tensor
        """
        error = x - x_reconstructed # Calculate difference between the two images
        error_squared = error ** 2 # Square? Why square? Because 1. It makes larger errors a really big deal, and 2. It removes negative values.
        mse_loss = torch.mean(error_squared) # Get mean squared error loss
        return mse_loss
    def forward(self, x):
        mu, log_var = self.encode(x) # Encode the input tensor to get mean and log variance
        z = self.reparameterize(mu, log_var) # Reparameterize to get latent variable z
        x_reconstructed = self.decode(z) # Decode z to get reconstructed tensor
        return x_reconstructed, mu, log_var


def main():

    vae = VAE(latent_dim=2, input_dim=128*128*3, hidden_dim=512) # 128x128 image with 3 channels (RGB)

    # Get all images paths
    image_paths = get_image_paths("data")
    print(image_paths)
    image_tensors = []
    for image_path in image_paths:
        # Load the image as a tensor
        tensor = load_image(image_path)
        print(f"Loaded image {image_paths.index(image_path) + 1} out of {len(image_paths)} tensors from {image_path} with shape {tensor.shape}")
        image_tensors.append(tensor)
    print(f"Loaded {len(image_tensors)} tensors from {len(image_paths)} images")

    
if __name__ == "__main__":
    main()