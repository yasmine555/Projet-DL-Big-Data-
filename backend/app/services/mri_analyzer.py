import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging
from io import BytesIO
import cv2

logger = logging.getLogger(__name__)


class MRIAnalyzer:
    """Service for analyzing brain MRI scans using TensorFlow model for Alzheimer's detection"""
    
    # Correct alphabetical mapping (standard Keras flow_from_directory)
    # 0: Mild Dementia, 1: Moderate Dementia, 2: Non Dementia, 3: Very Mild Dementia
    DEFAULT_CLASS_LABELS = {
        0: 'Mild Dementia',
        1: 'Moderate Dementia', 
        2: 'Non Dementia',
        3: 'Very Mild Dementia'
    }
      
    def __init__(self, model_path: str, class_labels: Optional[Dict[int, str]] = None):
        """
        Initialize MRI analyzer with model
        """
        self.model_path = Path(model_path)
        self.class_labels = class_labels or self.DEFAULT_CLASS_LABELS
        self.model = None
        self.image_size = (128, 128)
        
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model"""
        try:
            if not self.model_path.exists():
                logger.warning(f"MRI model not found at: {self.model_path}")
                return
            
            logger.info(f"Loading MRI model from {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info("MRI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MRI model: {e}")
            raise
    
    def preprocess_image(self, image_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for model input
        Returns (batch_array, original_array_for_xai)
        """
        try:
            # Load and convert to RGB
            img = Image.open(BytesIO(image_data)).convert("RGB")
            img = img.resize(self.image_size)
            
            # For model input: normalize to [0, 1]
            x = np.array(img, dtype=np.float32) / 255.0
            
            # For XAI/Visualization: keep original 0-255 uint8
            orig = np.array(img)
            
            return np.expand_dims(x, axis=0), orig
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Failed to preprocess image: {e}")

    def extract_brain_mask(self, img_array: np.ndarray) -> np.ndarray:
        """Simple brain mask extraction using thresholding"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask.astype(np.float32) / 255.0

    def compute_gradcam(self, model, img_batch, class_idx, layer_name="block5_conv3") -> np.ndarray:
        """Compute Grad-CAM heatmap for a given class"""
        try:
            # Check if layer exists, otherwise find last conv layer
            try:
                grad_model = tf.keras.models.Model(
                    [model.inputs], [model.get_layer(layer_name).output, model.output]
                )
            except ValueError:
                # Fallback to last conv layer if block5_conv3 not found (e.g. EfficientNet)
                last_conv_layer = next(l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D))
                grad_model = tf.keras.models.Model(
                    [model.inputs], [last_conv_layer.output, model.output]
                )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_batch)
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
        except Exception as e:
            logger.error(f"Grad-CAM computation failed: {e}")
            return np.zeros(self.image_size)

    def overlay_heatmap(self, img, heatmap, mask=None, alpha=0.4):
        """Overlay heatmap on original image"""
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        if mask is not None:
            heatmap_resized = heatmap_resized * mask

        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        return overlayed

    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze MRI image and return prediction with XAI results"""
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise RuntimeError("Model not loaded.")
        
        try:
            # Preprocess
            img_batch, orig_img = self.preprocess_image(image_data)
            
            # Predict
            preds = self.model.predict(img_batch, verbose=0)[0]
            class_idx = int(np.argmax(preds))
            confidence = float(preds[class_idx])
            
            # XAI: Grad-CAM
            heatmap = self.compute_gradcam(self.model, img_batch, class_idx)
            brain_mask = self.extract_brain_mask(orig_img)
            
            # Create visualization
            xai_img = self.overlay_heatmap(orig_img, heatmap, mask=brain_mask)
            
            # Compute attention ratio (brain vs total)
            total_attention = np.sum(heatmap)
            brain_attention = np.sum(heatmap * brain_mask)
            attention_ratio = float(brain_attention / total_attention) if total_attention > 0 else 0
            
            # Probabilities map
            prob_dict = {
                self.class_labels.get(i, f"Class {i}"): float(preds[i])
                for i in range(len(preds))
            }
            
            result = {
                "prediction_class": self.class_labels.get(class_idx, f"Class {class_idx}"),
                "confidence": confidence,
                "probabilities": prob_dict,
                "class_index": class_idx,
                "xai_image": xai_img,
                "brain_attention_ratio": attention_ratio
            }
            
            logger.info(f"MRI prediction: {result['prediction_class']} ({confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# Global instance
_mri_analyzer_instance: Optional[MRIAnalyzer] = None


def get_mri_analyzer() -> MRIAnalyzer:
    if _mri_analyzer_instance is None:
        raise RuntimeError("MRI analyzer not initialized.")
    return _mri_analyzer_instance


def initialize_mri_analyzer(model_path: str, class_labels: Optional[Dict[int, str]] = None):
    global _mri_analyzer_instance
    _mri_analyzer_instance = MRIAnalyzer(model_path, class_labels)
    return _mri_analyzer_instance

