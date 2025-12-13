"""
MRI Brain Analysis Service
Handles loading and running the brain MRI classification model
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from io import BytesIO

logger = logging.getLogger(__name__)


class MRIAnalyzer:
    """Service for analyzing brain MRI scans using TensorFlow model"""
    
    # Default class labels for brain tumor classification
    # User can override these if different
    DEFAULT_CLASS_LABELS = {
        0: "No Tumor",
        1: "Glioma",
        2: "Meningioma",
        3: "Pituitary Tumor"
    }
    
    def __init__(self, model_path: str, class_labels: Optional[Dict[int, str]] = None):
        """
        Initialize MRI analyzer with model
        
        Args:
            model_path: Path to the .h5 model file
            class_labels: Optional dict mapping class indices to labels
        """
        self.model_path = Path(model_path)
        self.class_labels = class_labels or self.DEFAULT_CLASS_LABELS
        self.model = None
        self.image_size = (128, 128)  # Model expects 128x128 images
        
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"MRI model not found at: {self.model_path}. "
                    f"Please ensure 'best_model.h5' is in the correct location."
                )
            
            logger.info(f"Loading MRI model from {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info("MRI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MRI model: {e}")
            raise
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Preprocessed numpy array ready for prediction
        """
        try:
            # Load image from bytes
            img = Image.open(BytesIO(image_data))
            
            # Convert to RGB (in case it's grayscale or RGBA)
            img = img.convert("RGB")
            
            # Resize to model's expected size
            img = img.resize(self.image_size)
            
            # Convert to numpy array and normalize to [0, 1]
            x = np.array(img, dtype=np.float32) / 255.0
            
            # Add batch dimension: (1, 128, 128, 3)
            x = np.expand_dims(x, axis=0)
            
            return x
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def predict(self, image_data: bytes) -> Dict[str, any]:
        """
        Analyze MRI image and return prediction
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dict containing:
                - prediction_class: Predicted class label
                - confidence: Confidence score (0-1)
                - probabilities: Dict of all class probabilities
                - class_index: Index of predicted class
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
        
        try:
            # Preprocess image
            x = self.preprocess_image(image_data)
            
            # Get prediction
            pred = self.model.predict(x, verbose=0)
            
            # Extract probabilities (first batch item)
            probabilities = pred[0]
            
            # Get predicted class (highest probability)
            class_index = int(np.argmax(probabilities))
            confidence = float(probabilities[class_index])
            
            # Map probabilities to class labels
            prob_dict = {
                self.class_labels.get(i, f"Class {i}"): float(probabilities[i])
                for i in range(len(probabilities))
            }
            
            result = {
                "prediction_class": self.class_labels.get(class_index, f"Class {class_index}"),
                "confidence": confidence,
                "probabilities": prob_dict,
                "class_index": class_index
            }
            
            logger.info(f"MRI prediction: {result['prediction_class']} (confidence: {confidence:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# Global instance (will be initialized in main.py)
_mri_analyzer_instance: Optional[MRIAnalyzer] = None


def get_mri_analyzer() -> MRIAnalyzer:
    """Get the global MRI analyzer instance"""
    if _mri_analyzer_instance is None:
        raise RuntimeError("MRI analyzer not initialized. Call initialize_mri_analyzer() first.")
    return _mri_analyzer_instance


def initialize_mri_analyzer(model_path: str, class_labels: Optional[Dict[int, str]] = None):
    """Initialize the global MRI analyzer instance"""
    global _mri_analyzer_instance
    _mri_analyzer_instance = MRIAnalyzer(model_path, class_labels)
    return _mri_analyzer_instance
