import nibabel as nib
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, root_dir, patient_ids):
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.image_files = []

        for patient_id in patient_ids:
            image_paths = [
                os.path.join(root_dir, f"BraTS20_Training_{patient_id:03d}", f"BraTS20_Training_{patient_id:03d}_{modality}.nii")
                for modality in ["flair", "t1", "t1ce", "t2"]
            ]
            self.image_files.append(image_paths)


    def __len__(self):
        num_images = sum([nib.load(image_paths[0]).shape[2] for image_paths in self.image_files])
        return num_images - num_images % 155  # Total number of images

    def __getitem__(self, idx):
        
        patient_idx = idx // 155
        image_idx = idx % 155
        
        image_paths = self.image_files[patient_idx]
                
        image_arrays = [np.array(nib.load(image_path).get_fdata())[:, :, image_idx] for image_path in image_paths]

        # Stack along the channel dimension
        image_array = np.stack(image_arrays)
        
        mask_path = image_paths[0].replace('_flair.nii', '_seg.nii')
        mask_array = np.array(nib.load(mask_path).get_fdata())[:, :, image_idx]

        # Convert to 3D (add channels dimension)
        image_tensor = torch.tensor(image_array)
        mask_tensor = torch.tensor(mask_array)
            
        
        mask_tensor = torch.clamp(mask_tensor, min=0, max=1)

        return {"img": image_tensor, "label": mask_tensor}


def load_data(start_patient_id, end_patient_id):
    dataset_path = "MICCAI_BraTS2020_TrainingData"

    all_patient_ids = list(range(start_patient_id, end_patient_id + 1))

    # Create datasets for training and testing
    train_dataset = CustomDataset(root_dir=dataset_path, patient_ids=all_patient_ids)

    # return train_dataset

    # Use DataLoader for batching
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    return trainloader