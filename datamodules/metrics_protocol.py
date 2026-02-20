"""
Metrics Protocol for DataModules.

This module defines the interface that datamodules should implement
to provide dataset-specific metric computation during validation/testing.

The goal is to keep train.py generic while allowing each datamodule to
define its own metrics (e.g., SWD/Energy/Feature-MMD for point clouds, FID for images).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class MetricsCapableDataModule(Protocol):
    """
    Protocol for datamodules that can compute metrics.
    
    DataModules implementing this protocol can be used by the generic
    validation/test loop in the LightningModule to compute dataset-specific metrics.
    """
    
    @property
    def is_metrics_capable(self) -> bool:
        """Return True if this datamodule supports metrics computation."""
        ...
    
    def get_eval_config(self) -> "EvalConfig":
        """
        Return evaluation configuration for this dataset.
        
        Returns:
            EvalConfig with class_ids, samples_per_class, sample_shape, etc.
        """
        ...
    
    def collect_real_samples_by_class(
        self,
        split: str,
        samples_per_class: int,
        batch_size: int = 64,
    ) -> Dict[int, np.ndarray]:
        """
        Collect real samples from the dataset, organized by class.
        
        Args:
            split: "train", "val", or "test"
            samples_per_class: number of samples to collect per class
            batch_size: batch size for loading
            
        Returns:
            Dict mapping class_id -> numpy array of samples [N, ...]
        """
        ...
    
    def compute_metrics(
        self,
        real_samples: Dict[int, np.ndarray],
        generated_samples: Dict[int, np.ndarray],
        split: str,
    ) -> Dict[str, float]:
        """
        Compute dataset-specific metrics comparing real and generated samples.
        
        Args:
            real_samples: Dict mapping class_id -> real samples array
            generated_samples: Dict mapping class_id -> generated samples array
            split: "val" or "test" (for metric naming)
            
        Returns:
            Dict of metric_name -> metric_value
        """
        ...


class EvalConfig:
    """Configuration for evaluation/metrics computation."""
    
    def __init__(
        self,
        class_ids: List[int],
        samples_per_class: int,
        sample_shape: Tuple[int, ...],
        num_classes: int,
        needs_decoding: bool = False,
    ):
        """
        Args:
            class_ids: List of class IDs to evaluate
            samples_per_class: Number of samples per class for metrics
            sample_shape: Shape of a single sample (excluding batch dim)
            num_classes: Total number of classes
            needs_decoding: Whether generated samples need VAE decoding
        """
        self.class_ids = class_ids
        self.samples_per_class = samples_per_class
        self.sample_shape = sample_shape
        self.num_classes = num_classes
        self.needs_decoding = needs_decoding


class BaseMetricsDataModule(ABC):
    """
    Abstract base class for datamodules that support metrics computation.
    
    Subclasses should implement the abstract methods to provide
    dataset-specific metric computation.
    """
    
    is_metrics_capable: bool = True
    
    @abstractmethod
    def get_eval_config(self) -> EvalConfig:
        """Return evaluation configuration."""
        raise NotImplementedError
    
    @abstractmethod
    def collect_real_samples_by_class(
        self,
        split: str,
        samples_per_class: int,
        batch_size: int = 64,
    ) -> Dict[int, np.ndarray]:
        """Collect real samples organized by class."""
        raise NotImplementedError
    
    @abstractmethod
    def compute_metrics(
        self,
        real_samples: Dict[int, np.ndarray],
        generated_samples: Dict[int, np.ndarray],
        split: str,
    ) -> Dict[str, float]:
        """Compute metrics comparing real and generated samples."""
        raise NotImplementedError
