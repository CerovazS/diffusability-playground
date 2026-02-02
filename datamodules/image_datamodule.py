from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import hydra

from datamodules.metrics_protocol import BaseMetricsDataModule, EvalConfig
from typing import Dict, List, Optional

 
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                                   Data Module                                #
#################################################################################

class ImageFolderDataModule(LightningDataModule, BaseMetricsDataModule):
    def __init__(
        self,
        data_path,
        image_size,
        global_batch_size,
        num_workers,
        pin_memory=True,
        drop_last=True,
        samples_per_class_for_metrics: Optional[int] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.num_workers = num_workers
        self.global_batch_size = global_batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.samples_per_class_for_metrics = samples_per_class_for_metrics
        self.dataset = None
        self.local_batch_size = None
        self.is_metrics_capable = samples_per_class_for_metrics is not None

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        data_path = hydra.utils.to_absolute_path(self.data_path)
        self.dataset = ImageFolder(data_path, transform=transform)

    def train_dataloader(self):
        world_size = getattr(self.trainer, "world_size", 1)
        if self.global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world size.")
        self.local_batch_size = int(self.global_batch_size // world_size)
        return DataLoader(
            self.dataset,
            batch_size=self.local_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    # -------------------------
    # Metrics Interface (BaseMetricsDataModule)
    # -------------------------

    def get_eval_config(self) -> EvalConfig:
        """Return evaluation configuration for image metrics."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before requesting eval config.")
        if self.samples_per_class_for_metrics is None:
            raise ValueError("samples_per_class_for_metrics must be set for image metrics.")
        
        # Get class information from ImageFolder
        class_ids = list(range(len(self.dataset.classes)))
        
        # For images with VAE: latent shape is (4, H/8, W/8)
        latent_size = self.image_size // 8
        
        return EvalConfig(
            class_ids=class_ids,
            samples_per_class=self.samples_per_class_for_metrics,
            sample_shape=(4, latent_size, latent_size),  # VAE latent shape
            num_classes=len(class_ids),
            needs_decoding=True,  # Images need VAE decoding
        )

    def collect_real_samples_by_class(
        self,
        split: str,
        samples_per_class: int,
        batch_size: int = 64,
    ) -> Dict[int, np.ndarray]:
        """
        Collect real images from the dataset, organized by class.
        
        Args:
            split: Currently only "train" is supported for ImageFolder
            samples_per_class: Number of samples to collect per class
            batch_size: Batch size for loading
            
        Returns:
            Dict mapping class_id -> numpy array of shape [N, C, H, W]
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before collecting samples.")
        
        class_ids = list(range(len(self.dataset.classes)))
        images: Dict[int, List[np.ndarray]] = {cid: [] for cid in class_ids}
        counts: Dict[int, int] = {cid: 0 for cid in class_ids}
        
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        for xb, yb in loader:
            for i in range(xb.size(0)):
                y = int(yb[i].item())
                if counts[y] >= samples_per_class:
                    continue
                images[y].append(xb[i].cpu().numpy())
                counts[y] += 1
            if all(c >= samples_per_class for c in counts.values()):
                break

        return {k: np.stack(v, axis=0) for k, v in images.items() if len(v) > 0}

    def compute_metrics(
        self,
        real_samples: Dict[int, np.ndarray],
        generated_samples: Dict[int, np.ndarray],
        split: str,
    ) -> Dict[str, float]:
        """
        Compute image metrics.
        
        TODO: Implement FID, IS, or other image-specific metrics.
        For now returns empty dict - can be extended with:
        - FID (Fréchet Inception Distance)
        - IS (Inception Score)
        - LPIPS
        - SSIM
        
        Args:
            real_samples: Dict mapping class_id -> real images [N, C, H, W]
            generated_samples: Dict mapping class_id -> generated images [N, C, H, W]
            split: "val" or "test" for metric naming
            
        Returns:
            Dict of metric_name -> metric_value (empty for now)
        """
        # Placeholder for future image metrics implementation
        # Example implementation would look like:
        # 
        # from torchmetrics.image.fid import FrechetInceptionDistance
        # fid = FrechetInceptionDistance(feature=2048)
        # ...
        
        metrics: Dict[str, float] = {}
        
        # Basic statistics as placeholder
        for class_id in real_samples.keys():
            if class_id not in generated_samples:
                continue
            real = real_samples[class_id]
            gen = generated_samples[class_id]
            
            # Mean pixel difference (very basic, just for demonstration)
            # Replace with FID/IS in production
            mean_diff = float(np.abs(real.mean() - gen.mean()))
            metrics[f"{split}/mean_pixel_diff/class_{class_id}"] = mean_diff
        
        return metrics

