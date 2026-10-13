import torch
import torch.nn.functional as F
import numpy as np
import cv2

class StegoGradCAM:
    """
    Grad-CAM implementation designed specifically for the Sudarshana Fusion Model.
    It hooks into the deepest convolutional layer of Branch A (Pixel CNN) to 
    visualize where the network detects steganographic noise.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Register hooks to capture activations and gradients
        self._forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)

    def close(self):
        """Remove hooks so repeated Streamlit reruns do not retain references."""
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __del__(self):
        # Best-effort cleanup for callers that do not explicitly call close().
        try:
            self.close()
        except (AttributeError, RuntimeError):
            pass

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=1):
        """
        Generates the Grad-CAM heatmap for a given input tensor.
        target_class = 1 (Stego class by default)
        """
        # A perturbed tensor may be non-leaf (for example after torch.clamp),
        # so assigning requires_grad directly can raise. Always make an
        # independent leaf tensor for the explanation pass.
        input_tensor = input_tensor.detach().clone().requires_grad_(True)
        
        # Forward pass
        self.model.zero_grad()
        logits = self.model(input_tensor)
        
        # If target_class is None, use the model's prediction
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        # Get the score for the target class
        score = logits[0, target_class]
        
        # Backward pass to calculate gradients
        score.backward()
        
        # Get the gradients and activations captured by the hooks
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Global average pooling on gradients to get the weights for each channel
        weights = np.mean(gradients, axis=(1, 2))
        
        # Multiply each activation channel by its corresponding weight
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # Apply ReLU to keep only features that positively contribute to the class
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize the heatmap between 0 and 1
        if np.max(heatmap) == 0:
            return np.zeros_like(heatmap)
        heatmap /= np.max(heatmap)
        
        return heatmap

    @staticmethod
    def overlay_heatmap(image_np, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlays the generated heatmap onto the original image.
        """
        # Resize heatmap to match the original image size
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        
        # Convert heatmap to RGB heatmap image
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay the heatmap on the original image
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
        
        return overlay, heatmap_color
