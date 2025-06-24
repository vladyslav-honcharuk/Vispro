#!/usr/bin/env python3
"""
Vispro Image Processing Script
Usage: python vispro_replace.py <image_path>
Processes the image and replaces it with the inpainted version.
"""

import argparse
import sys
import os
from PIL import Image
import numpy as np

# Import Vispro modules
try:
    from src.utils import get_image_var
    from src.main_pipeline import get_marker_mask, get_inpainting_result
except ImportError as e:
    print(f"Error: Could not import Vispro modules: {e}")
    print("Make sure you're running this script from the Vispro directory")
    sys.exit(1)

def process_and_replace_image(image_path):
    """
    Process image with Vispro and replace original with inpainted version
    
    Args:
        image_path (str): Path to the input image
    """
    # Validate input file
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist")
        return False
    
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
        print(f"Error: Unsupported file format. Use PNG, JPG, JPEG, TIFF, or TIF")
        return False
    
    print(f"Processing image: {image_path}")
    
    try:
        # Get original image info for resolution preservation
        original_pil = Image.open(image_path)
        original_size = original_pil.size  # (width, height)
        original_mode = original_pil.mode
        print(f"Original image: {original_size[0]}x{original_size[1]}, mode: {original_mode}")
        
        # Run Vispro pipeline (only the essential parts)
        print("🔍 Detecting fiducial markers...")
        img_var, img_np = get_image_var(image_path)
        
        print("🎯 Generating marker mask...")
        marker_mask = get_marker_mask(img_var)
        
        print("🎨 Inpainting image (removing markers)...")
        inpainted_image = get_inpainting_result(img_np, marker_mask)
        
        # Ensure inpainted image matches original resolution
        inpainted_pil = Image.fromarray(inpainted_image)
        
        # Resize if dimensions don't match (shouldn't happen, but safety check)
        if inpainted_pil.size != original_size:
            print(f"Error: Images are not the same resolution {e}")
            print(f"⚠️  Resizing from {inpainted_pil.size} to {original_size}")
            inpainted_pil = inpainted_pil.resize(original_size, Image.LANCZOS)
        
        # Convert to original mode if necessary
        if inpainted_pil.mode != original_mode:
            if original_mode == 'L':  # Grayscale
                inpainted_pil = inpainted_pil.convert('L')
            elif original_mode == 'RGBA':
                inpainted_pil = inpainted_pil.convert('RGBA')
            elif original_mode == 'RGB':
                inpainted_pil = inpainted_pil.convert('RGB')
        
        # Create backup of original (optional - uncomment if you want backup)
        # backup_path = f"{os.path.splitext(image_path)[0]}_original{os.path.splitext(image_path)[1]}"
        # original_pil.save(backup_path)
        # print(f"💾 Original saved as: {backup_path}")
        
        # Replace original file with inpainted version
        inpainted_pil.save(image_path, quality=95, optimize=True)
        
        print(f"✅ Successfully processed and replaced: {image_path}")
        print(f"📏 Final image: {inpainted_pil.size[0]}x{inpainted_pil.size[1]}, mode: {inpainted_pil.mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process image with Vispro and replace original with inpainted version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vispro_replace.py image.png
  python vispro_replace.py /path/to/tissue_hires_image.png
  python vispro_replace.py "my image with spaces.jpg"
        """
    )
    
    parser.add_argument(
        'image_path',
        help='Path to the input image file'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of original image before replacement'
    )
    
    args = parser.parse_args()
    
    # Handle backup option
    if args.backup:
        try:
            backup_path = f"{os.path.splitext(args.image_path)[0]}_original{os.path.splitext(args.image_path)[1]}"
            original_img = Image.open(args.image_path)
            original_img.save(backup_path)
            print(f"💾 Backup created: {backup_path}")
        except Exception as e:
            print(f"⚠️  Could not create backup: {e}")
    
    # Process the image
    success = process_and_replace_image(args.image_path)
    
    if success:
        print("\n🎉 Processing complete!")
        sys.exit(0)
    else:
        print("\n💥 Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
