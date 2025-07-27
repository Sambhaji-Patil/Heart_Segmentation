import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import pyvista as pv
from skimage import measure
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- UNet3D Model (same as training) ---
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(self.conv_block(in_channels, feature))
            in_channels = feature

        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.conv_block(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = self.decoder[idx + 1](torch.cat((skip, x), dim=1))
        return self.final_conv(x)

class HeartInferenceComparison:
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = UNet3D().to(self.device)
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load trained model weights"""
        logger.info(f"Loading model from {model_path}")
        if model_path.endswith('.pth'):
            # Handle both direct state dict and checkpoint formats
            checkpoint = torch.load(model_path, map_location=self.device,weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def preprocess_image(self, img_path):
        """Preprocess input image"""
        logger.info(f"Loading image: {img_path}")
        img = nib.load(img_path)
        img_data = img.get_fdata().astype(np.float32)
        
        # Store original properties for later use
        self.original_shape = img_data.shape
        self.affine = img.affine
        self.header = img.header
        
        # Normalize image (same as training)
        img_data = np.clip(img_data, -1000, 1000)
        img_data = (img_data + 1000) / 2000.0
        
        # Add batch and channel dimensions
        img_tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0).to(self.device)
        return img_tensor, img_data
    
    def load_ground_truth(self, gt_path):
        """Load ground truth segmentation"""
        logger.info(f"Loading ground truth: {gt_path}")
        gt_img = nib.load(gt_path)
        gt_data = gt_img.get_fdata().astype(np.int64)
        return gt_data
    
    def sliding_window_inference(self, img_tensor, window_size=(96, 96, 96), overlap=0.5):
        """Perform sliding window inference for large volumes"""
        _, _, d, h, w = img_tensor.shape
        wd, wh, ww = window_size
        
        # Calculate step sizes based on overlap
        step_d = int(wd * (1 - overlap))
        step_h = int(wh * (1 - overlap))
        step_w = int(ww * (1 - overlap))
        
        # Initialize output
        output = torch.zeros((1, 2, d, h, w), device=self.device)
        count_map = torch.zeros((1, 1, d, h, w), device=self.device)
        
        logger.info("Performing sliding window inference...")
        
        # Sliding window inference
        for z in range(0, d - wd + 1, step_d):
            for y in range(0, h - wh + 1, step_h):
                for x in range(0, w - ww + 1, step_w):
                    # Extract patch
                    patch = img_tensor[:, :, z:z+wd, y:y+wh, x:x+ww]
                    
                    # Inference
                    with torch.no_grad():
                        pred = self.model(patch)
                    
                    # Add to output
                    output[:, :, z:z+wd, y:y+wh, x:x+ww] += pred
                    count_map[:, :, z:z+wd, y:y+wh, x:x+ww] += 1
        
        # Handle borders
        if d % step_d != 0:
            z = d - wd
            for y in range(0, h - wh + 1, step_h):
                for x in range(0, w - ww + 1, step_w):
                    patch = img_tensor[:, :, z:z+wd, y:y+wh, x:x+ww]
                    with torch.no_grad():
                        pred = self.model(patch)
                    output[:, :, z:z+wd, y:y+wh, x:x+ww] += pred
                    count_map[:, :, z:z+wd, y:y+wh, x:x+ww] += 1
        
        # Average overlapping predictions
        output = output / count_map
        return output
    
    def postprocess_prediction(self, prediction):
        """Convert prediction to segmentation mask"""
        # Apply softmax and get argmax
        prediction = F.softmax(prediction, dim=1)
        segmentation = torch.argmax(prediction, dim=1)
        
        # Convert to numpy
        segmentation = segmentation.cpu().numpy().squeeze()
        return segmentation
    
    def create_3d_mesh(self, segmentation, spacing=(1.0, 1.0, 1.0), smoothing_iterations=100):
        """Create 3D mesh from segmentation using marching cubes"""
        logger.info("Creating 3D mesh from prediction...")
        
        # Extract left atrium (label = 1)
        atrium_mask = (segmentation == 1).astype(np.uint8)
        
        if np.sum(atrium_mask) == 0:
            logger.warning("No left atrium found in segmentation!")
            return None
        
        # Use marching cubes to create mesh
        try:
            vertices, faces, _, _ = measure.marching_cubes(
                atrium_mask, 
                level=0.5, 
                spacing=spacing,
                allow_degenerate=False
            )
            
            # Create PyVista mesh
            faces_with_count = np.column_stack([
                np.full(faces.shape[0], 3),  # All triangles
                faces
            ]).flatten()
            
            mesh = pv.PolyData(vertices, faces_with_count)
            
            # Smooth the mesh
            if smoothing_iterations > 0:
                mesh = mesh.smooth(n_iter=smoothing_iterations, relaxation_factor=0.1)
            
            logger.info(f"Prediction mesh created with {mesh.n_points} vertices and {mesh.n_cells} faces")
            return mesh
            
        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            return None
    
    def create_volume_with_overlays(self, original_image, ground_truth, prediction, downsample_factor=2):
        """Create volume data with ground truth and prediction overlays"""
        logger.info("Creating volume with overlays...")
        
        # Downsample all data for performance
        original_ds = original_image[::downsample_factor, ::downsample_factor, ::downsample_factor]
        gt_ds = ground_truth[::downsample_factor, ::downsample_factor, ::downsample_factor]
        pred_ds = prediction[::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        # Create base volume for original image
        vol_original = pv.ImageData(dimensions=original_ds.shape)
        vol_original.point_data["intensity"] = original_ds.flatten(order="F")
        
        # Create volume for ground truth overlay (only atrium voxels)
        gt_atrium = (gt_ds == 1).astype(np.float32)
        vol_gt = pv.ImageData(dimensions=gt_atrium.shape)
        vol_gt.point_data["ground_truth"] = gt_atrium.flatten(order="F")
        
        # Create volume for prediction overlay (only atrium voxels)
        pred_atrium = (pred_ds == 1).astype(np.float32)
        vol_pred = pv.ImageData(dimensions=pred_atrium.shape)
        vol_pred.point_data["prediction"] = pred_atrium.flatten(order="F")
        
        return vol_original, vol_gt, vol_pred
    
    def visualize_comparison_3d(self, original_image, ground_truth, prediction, extracted_mesh, save_path=None):
        """Visualize comparison with in-body overlays and extracted mesh"""
        logger.info("Creating comprehensive 3D visualization...")
        
        # Create plotter with larger window
        plotter = pv.Plotter(window_size=(1600, 900))
        
        # Create volumes with overlays
        vol_original, vol_gt, vol_pred = self.create_volume_with_overlays(original_image, ground_truth, prediction)
        
        # Add original image as semi-transparent volume
        try:
            plotter.add_volume(
                vol_original,
                opacity=[0, 0.05, 0.1, 0.15, 0.2],  # Very subtle background
                cmap='gray',
                name='original'
            )
            logger.info("Added original volume")
        except Exception as e:
            logger.warning(f"Could not add original volume: {e}")
        
        # Add ground truth overlay in blue
        try:
            plotter.add_volume(
                vol_gt,
                opacity=[0, 0.7],  # 0 for background, 0.7 for atrium
                cmap='Blues',
                name='ground_truth'
            )
            logger.info("Added ground truth overlay in blue")
        except Exception as e:
            logger.warning(f"Could not add ground truth overlay: {e}")
        
        # Add prediction overlay in red (semi-transparent to see overlap)
        try:
            plotter.add_volume(
                vol_pred,
                opacity=[0, 0.6],  # 0 for background, 0.6 for atrium
                cmap='reds',
                name='prediction'
            )
            logger.info("Added prediction overlay in red")
        except Exception as e:
            logger.warning(f"Could not add prediction overlay: {e}")
        
        # Add extracted mesh (shifted to the side for separate viewing)
        if extracted_mesh is not None:
            # Shift mesh to the right of the original volume
            bounds = vol_original.bounds
            shift_x = (bounds[1] - bounds[0]) * 1.5  # Move 1.5 times the width to the right
            
            shifted_mesh = extracted_mesh.translate([shift_x, 0, 0])
            
            plotter.add_mesh(
                shifted_mesh,
                color='red',
                opacity=0.8,
                smooth_shading=True,
                name='extracted_prediction'
            )
            
            # Add wireframe for better detail
            plotter.add_mesh(
                shifted_mesh,
                style='wireframe',
                color='darkred',
                opacity=0.4,
                line_width=1,
                name='extracted_wireframe'
            )
            
            logger.info("Added extracted mesh (shifted)")
        
        # Add title and labels
        plotter.add_title("Heart Left Atrium: Ground Truth (Blue) vs Prediction (Red) + Extracted Mesh", font_size=14)
        
        # Add text annotations
        plotter.add_text("Original Volume with Overlays", position='upper_left', font_size=12)
        plotter.add_text("Blue = Ground Truth\nRed = Prediction", position='lower_left', font_size=10)
        plotter.add_text("Extracted Prediction →", position='upper_right', font_size=12)
        
        # Set camera for good overview
        plotter.camera_position = 'iso'
        
        # Add axes
        plotter.add_axes()
        
        # Save screenshot if requested (off-screen mode)
        if save_path:
            logger.info(f"Saving comparison visualization to {save_path}")
            plotter_offscreen = pv.Plotter(off_screen=True, window_size=(1600, 900))
            
            # Recreate scene for off-screen rendering
            try:
                plotter_offscreen.add_volume(vol_original, opacity=[0, 0.05, 0.1, 0.15, 0.2], cmap='gray')
                plotter_offscreen.add_volume(vol_gt, opacity=[0, 0.7], cmap='Blues')
                plotter_offscreen.add_volume(vol_pred, opacity=[0, 0.6], cmap='reds')
                
                if extracted_mesh is not None:
                    bounds = vol_original.bounds
                    shift_x = (bounds[1] - bounds[0]) * 1.5
                    shifted_mesh = extracted_mesh.translate([shift_x, 0, 0])
                    plotter_offscreen.add_mesh(shifted_mesh, color='red', opacity=0.8, smooth_shading=True)
                    plotter_offscreen.add_mesh(shifted_mesh, style='wireframe', color='darkred', opacity=0.4, line_width=1)
                
                plotter_offscreen.add_title("Heart Left Atrium: Ground Truth (Blue) vs Prediction (Red) + Extracted Mesh", font_size=14)
                plotter_offscreen.camera_position = 'iso'
                plotter_offscreen.screenshot(save_path, transparent_background=True)
                plotter_offscreen.close()
            except Exception as e:
                logger.warning(f"Could not save screenshot: {e}")
        
        # Show interactive plot
        plotter.show()
    
    def save_segmentation(self, segmentation, output_path):
        """Save segmentation as NIfTI file"""
        logger.info(f"Saving segmentation to {output_path}")
        
        # Create NIfTI image with original affine and header
        seg_img = nib.Nifti1Image(
            segmentation.astype(np.int16), 
            affine=self.affine, 
            header=self.header
        )
        nib.save(seg_img, output_path)
    
    def run_comparison_inference(self, img_path, gt_path, output_dir=None, visualize=True, save_mesh=False):
        """Complete inference pipeline with comparison visualization"""
        logger.info("Starting comparison inference pipeline...")
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.dirname(img_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        
        # Load data
        img_tensor, original_image = self.preprocess_image(img_path)
        ground_truth = self.load_ground_truth(gt_path)
        
        # Inference
        if img_tensor.shape[2:] != (96, 96, 96):
            # Use sliding window for large images
            prediction_logits = self.sliding_window_inference(img_tensor)
        else:
            # Direct inference for patch-sized images
            with torch.no_grad():
                prediction_logits = self.model(img_tensor)
        
        # Postprocess
        prediction = self.postprocess_prediction(prediction_logits)
        
        # Save segmentation
        seg_path = os.path.join(output_dir, f"{base_name}_prediction.nii.gz")
        self.save_segmentation(prediction, seg_path)
        
        # Create 3D mesh from prediction
        extracted_mesh = self.create_3d_mesh(prediction)
        
        # Save mesh if requested
        if save_mesh and extracted_mesh is not None:
            mesh_path = os.path.join(output_dir, f"{base_name}_extracted_atrium.stl")
            extracted_mesh.save(mesh_path)
            logger.info(f"Extracted mesh saved to {mesh_path}")
        
        # Create comparison visualization
        if visualize:
            screenshot_path = os.path.join(output_dir, f"{base_name}_comparison_3d.png")
            
            self.visualize_comparison_3d(
                original_image=original_image,
                ground_truth=ground_truth,
                prediction=prediction,
                extracted_mesh=extracted_mesh,
                save_path=screenshot_path
            )
        
        return prediction, extracted_mesh, ground_truth

def main():
    parser = argparse.ArgumentParser(description='Heart Left Atrium 3D Inference with Comparison')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input NIfTI image')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth NIfTI label')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as input)')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip 3D visualization')
    parser.add_argument('--save_mesh', action='store_true',
                       help='Save 3D mesh as STL file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = HeartInferenceComparison(args.model_path, args.device)
    
    # Run comparison inference
    prediction, mesh, ground_truth = inference.run_comparison_inference(
        args.image_path,
        args.gt_path,
        args.output_dir,
        visualize=not args.no_visualize,
        save_mesh=args.save_mesh
    )
    
    logger.info("Comparison inference completed successfully!")

if __name__ == '__main__':
    main()