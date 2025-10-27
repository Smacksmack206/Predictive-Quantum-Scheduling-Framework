#!/usr/bin/env python3
"""
Create PQS Framework App Icon
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    """Create a simple but professional app icon"""
    # Create a 1024x1024 image (standard macOS icon size)
    size = 1024
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw gradient background circle
    center = size // 2
    radius = int(size * 0.45)
    
    # Create a nice quantum-themed gradient
    for i in range(radius, 0, -2):
        # Purple to blue gradient
        ratio = i / radius
        r = int(80 + (150 - 80) * (1 - ratio))
        g = int(50 + (100 - 50) * (1 - ratio))
        b = int(200 + (255 - 200) * (1 - ratio))
        
        draw.ellipse(
            [center - i, center - i, center + i, center + i],
            fill=(r, g, b, 255)
        )
    
    # Draw atom symbol (simplified)
    # Nucleus
    nucleus_radius = int(radius * 0.15)
    draw.ellipse(
        [center - nucleus_radius, center - nucleus_radius,
         center + nucleus_radius, center + nucleus_radius],
        fill=(255, 255, 255, 255)
    )
    
    # Electron orbits
    orbit_width = 8
    orbit_radius = int(radius * 0.7)
    
    # Horizontal orbit
    draw.ellipse(
        [center - orbit_radius, center - orbit_width,
         center + orbit_radius, center + orbit_width],
        outline=(255, 255, 255, 200),
        width=orbit_width
    )
    
    # Diagonal orbit 1
    draw.ellipse(
        [center - orbit_radius, center - orbit_radius,
         center + orbit_radius, center + orbit_radius],
        outline=(255, 255, 255, 150),
        width=orbit_width
    )
    
    # Diagonal orbit 2
    draw.ellipse(
        [center - orbit_radius, center - orbit_radius,
         center + orbit_radius, center + orbit_radius],
        outline=(255, 255, 255, 150),
        width=orbit_width
    )
    
    # Draw electrons
    electron_radius = int(radius * 0.08)
    positions = [
        (center + orbit_radius, center),
        (center - orbit_radius, center),
        (center, center + orbit_radius),
    ]
    
    for x, y in positions:
        draw.ellipse(
            [x - electron_radius, y - electron_radius,
             x + electron_radius, y + electron_radius],
            fill=(100, 255, 255, 255)
        )
    
    # Save as PNG
    img.save('pqs_icon.png', 'PNG')
    print("✅ Created pqs_icon.png")
    
    # Create smaller sizes for different uses
    for size in [512, 256, 128, 64, 32, 16]:
        small = img.resize((size, size), Image.Resampling.LANCZOS)
        small.save(f'pqs_icon_{size}.png', 'PNG')
    
    print("✅ Created icon set")
    
    # Try to create .icns file (macOS icon format)
    try:
        # Create iconset directory
        os.makedirs('pqs_icon.iconset', exist_ok=True)
        
        # Save required sizes
        sizes = {
            16: ['icon_16x16.png'],
            32: ['icon_16x16@2x.png', 'icon_32x32.png'],
            64: ['icon_32x32@2x.png'],
            128: ['icon_128x128.png'],
            256: ['icon_128x128@2x.png', 'icon_256x256.png'],
            512: ['icon_256x256@2x.png', 'icon_512x512.png'],
            1024: ['icon_512x512@2x.png']
        }
        
        for size, filenames in sizes.items():
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            for filename in filenames:
                resized.save(f'pqs_icon.iconset/{filename}', 'PNG')
        
        # Convert to icns using iconutil (macOS only)
        import subprocess
        result = subprocess.run(
            ['iconutil', '-c', 'icns', 'pqs_icon.iconset', '-o', 'pqs_icon.icns'],
            capture_output=True
        )
        
        if result.returncode == 0:
            print("✅ Created pqs_icon.icns")
        else:
            print("⚠️ Could not create .icns file (iconutil not available)")
            
    except Exception as e:
        print(f"⚠️ Could not create .icns file: {e}")

if __name__ == "__main__":
    create_icon()
