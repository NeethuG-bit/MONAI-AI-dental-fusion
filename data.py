import numpy as np
from scipy import ndimage

def generate_panoramic(size=(512, 256)):
  img = np.zeros(size, dtype=np.float32)
  center_y = size[1]//2

  # Jawbone
  for x in range(size[0]):
    y_offset = int(35 * np.sin(x / size[0] * np.pi))
    y_pos = center_y + y_offset
    for dy in range(-15, 15):
      if 0 <= y_pos + dy < size[1]:
        img[x, y_pos + dy] = 0.6

  # Teeth (32 teeth)
  for i in range(32):
    x_pos = i * size[0] // 32 + size[0] // 64
    y_offset = int(35 * np.sin(x_pos / size[0] * np.pi))
    y_pos = center_y + y_offset

    # Crown
    for dx in range(-6, 6):
      for dy in range(-17, 0):
        x, y = x_pos + dx, y_pos + dy
        if 0 <= x < size[0] and 0 <= y < size[1]:
          if x**2 + (dy*0.5)**2 < 36:
            img[x, y] = 1.0   # Enamel

    # Root
    for dx in range(-4, 4):
      for dy in range(0, 12):
        x, y = x_pos + dx, y_pos + dy
        if 0 <= x < size[0] and 0 <= y < size[1]:
          if dx**2 + dy**2 < 16:
            img[x, y] = 0.75   # Dentin

    # Caries (pathology) on some teeth
    if i % 7 == 0:
      for dx in range(-3, 3):
        for dy in range(-10, -5):
          x, y = x_pos + dx, y_pos + dy
          if 0 <= x < size[0] and 0 <= y < size[1]:
            if dx**2 + dy**2 < 6:
              img[x, y] *= 0.5

  img += np.random.normal(0, 0.02, size).astype(np.float32)
  img = ndimage.gaussian_filter(img, sigma=1.2)
  return np.clip(img, 0, 1) * 2000 - 1000

def generate_cbct(size=(128, 128, 64)):
  volume = np.zeros(size, dtype=np.float32)
  cx, cy, cz = size[0]//2, size[1]//2, size[2]//2

  # Mandible structure
  for z in range(size[2]):
    for theta in np.linspace(-2.5, 0, 150):
      radius = 50 + (z - cz) * 0.1
      x = int(cx + radius * np.cos(theta))
      y = int(cy + radius * 0.6 * np.sin(theta) + 30)

      if 0 <= x < size[0] and 0 <= y < size[1]:
        for r in range(10):
          for t in np.linspace(0, 6.28, 25):
            dx, dy = int(r * np.cos(t)), int(r * np.sin(t))
            if 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
              if r < 3:
                volume[x+dx, y+dy, z] = 1200
              else:
                volume[x+dx, y+dy, z] = 600

  # Teeth
  for i in range(14):
    theta = -2.5 + i * 2.5 / 14
    x = int(cx + 48 * np.cos(theta))
    y = int(cy + 48 * 0.6 * np.sin(theta) + 30)

    for z in range(max(0, cz-15), min(size[2], cz+20)):
      for dx in range(-5, 5):
        for dy in range(-5, 5):
          if 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 4:
              volume[x+dx, y+dy, z] = 1600 if z < cz else 1200

    # Periapical lesian (pathology)
    if i % 7 == 0:
            lesion_z = min(cz + 22, size[2] - 1)
            for dx in range(-6, 6):
                for dy in range(-6, 6):
                    if 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                        if dx**2 + dy**2 < 30:
                            volume[x+dx, y+dy, lesion_z] = -200

    # Maxillary sinus
    for x in range(cx-30, cx+30):
        for y in range(cy-70, cy-40):
            for z in range(cz-25, cz+25):
                if 0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]:
                    dx, dy, dz = x-cx, y-(cy-55), z-cz
                    if dx**2/900 + dy**2/400 + dz**2/625 < 1:
                        volume[x, y, z] = -900

    volume += np.random.normal(0, 20, size).astype(np.float32)
    volume = ndimage.gaussian_filter(volume, sigma=0.8)
    return np.clip(volume, -1000, 3000)

def generate_soft_tissue(size=(128, 128)):
    """Generate soft tissue surface"""
    img = np.zeros(size, dtype=np.float32)
    cx, cy = size[0]//2, size[1]//2

    # Gingival contours
    for theta in np.linspace(0, 6.28, 300):
        scallop = 5 * np.sin(12 * theta)
        radius = 50 + scallop
        x = int(cx + radius * np.cos(theta))
        y = int(cy + radius * 0.7 * np.sin(theta))

        if 0 <= x < size[0] and 0 <= y < size[1]:
            for r in range(20):
                for t in np.linspace(0, 6.28, 30):
                    dx, dy = int(r * np.cos(t)), int(r * np.sin(t))
                    if 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                        img[x+dx, y+dy] = max(img[x+dx, y+dy], 8 * (1 - r/20))

    # Teeth surfaces
    for i in range(14):
        theta = i * 6.28 / 14
        x = int(cx + 48 * np.cos(theta))
        y = int(cy + 48 * 0.7 * np.sin(theta))

        for dx in range(-8, 8):
            for dy in range(-8, 8):
                if 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 6:
                        img[x+dx, y+dy] = max(img[x+dx, y+dy], 12 * (1 - dist/6))

    img = ndimage.gaussian_filter(img, sigma=1.5)
    return np.clip(img, 0, 15)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Generating Clinical-Grade Synthetic Data...")
    print("=" * 80)

    panoramic = generate_panoramic()
    cbct = generate_cbct()
    soft_tissue = generate_soft_tissue()

    print(f"\n Panoramic: {panoramic.shape}, HU range: [{panoramic.min():.0f}, {panoramic.max():.0f}]")
    print(f" CBCT: {cbct.shape}, HU range: [{cbct.min():.0f}, {cbct.max():.0f}]")
    print(f" Soft tissue: {soft_tissue.shape}, depth range: [{soft_tissue.min():.1f}, {soft_tissue.max():.1f}] mm")