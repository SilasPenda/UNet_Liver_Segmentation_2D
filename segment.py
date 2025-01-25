import os
import nibabel as nib
import numpy as np
import argparse
from PIL import Image

import torch
import torchvision.transforms.functional as TF

from src.model import UNet2D


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])


def main():
    parser = argparse.ArgumentParser(description='3D Liver Segmentation')
    parser.add_argument('-i', '--input_dir', type=str, help='Input directory', required=True)
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet2D(input_channels=3, n_classes=3)
    model.to(device)

    load_checkpoint(torch.load("best.pth", map_location=torch.device(device)), model)

    for file_name in os.listdir(args.input_dir):
        if file_name.endswith(".nii.gz"):
            # Load NIfTI image
            nifti_image = nib.load(os.path.join(args.input_dir, file_name))
            original_affine = nifti_image.affine
            image = nifti_image.get_fdata()

            segmented_slices = []

            # Iterate through slices along the z-axis
            for i in range(image.shape[-1]):  # Loop over the third dimension (Z-axis)
                slice_2d = image[:, :, i]  # Extract the i-th 2D slice

                # Convert the slice to RGB and then to a tensor
                slice_2d_rgb = Image.fromarray(slice_2d).convert("RGB")
                slice_2d_tensor = TF.to_tensor(np.array(slice_2d_rgb)).float().to(device)

                # Perform model inference
                with torch.no_grad():
                    prediction = model(slice_2d_tensor.unsqueeze(0))  # Add batch dimension

                # Get the predicted class for each pixel
                segmented_slice = torch.argmax(prediction, dim=1).squeeze(0)  # Remove batch dimension

                # Convert to NumPy array
                segmented_slice_np = segmented_slice.cpu().numpy()

                # Ensure data type compatibility
                segmented_slice_np = segmented_slice_np.astype(np.float32)

                segmented_slices.append(segmented_slice_np)

            # Stack the segmented slices along the z-axis to form a 3D volume
            segmented_volume = np.stack(segmented_slices, axis=-1).astype(np.int16)

            # Save the segmented volume as a NIfTI file
            output_path = os.path.join(args.output_dir, file_name)
            nib.save(nib.Nifti1Image(segmented_volume, affine=original_affine), output_path)

            print(f"Volume {file_name} Segmented successfully")



if __name__ == "__main__":
    main()