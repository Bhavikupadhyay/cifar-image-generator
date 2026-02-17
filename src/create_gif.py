import os
import re
from PIL import Image

def main():
    # Setup paths
    root_dir = os.getcwd()
    samples_dir = os.path.join(root_dir, 'assets', 'training_samples')
    output_path = os.path.join(root_dir, 'assets', 'training_evolution.gif')
    
    if not os.path.exists(samples_dir):
        print(f"Directory not found: {samples_dir}")
        return

    # List files and filter
    files = [f for f in os.listdir(samples_dir) if f.startswith('epoch_') and f.endswith('.png')]
    
    # Sort files by epoch number
    def extract_epoch(filename):
        match = re.search(r'epoch_(\d+)', filename)
        return int(match.group(1)) if match else 0

    files.sort(key=extract_epoch)
    
    if not files:
        print("No training sample images found.")
        return

    print(f"Found {len(files)} images. Creating GIF...")
    
    # Load images
    frames = []
    for f in files:
        img_path = os.path.join(samples_dir, f)
        frames.append(Image.open(img_path))
    
    # Save as GIF
    # Duration is in milliseconds. 200ms = 5 fps.
    frames[0].save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=200,
        loop=0
    )
    
    print(f"GIF saved to {output_path}")

if __name__ == '__main__':
    main()
