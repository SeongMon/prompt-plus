import os
from PIL import Image, ImageDraw, ImageFont
import math

# 이미지 폴더 경로
image_dir = "/mnt/ssd1/seongmin/textual_inversion/output_images/cat_statue_/3000"
output_path = "cat_statue_grid_with_labels.png"

# 2. PNG 이미지 파일 경로 수집
image_paths = sorted([
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.lower().endswith(".png")
])

# 3. 그리드 크기 결정
num_images = len(image_paths)
grid_cols = 5
grid_rows = math.ceil(num_images / grid_cols)

# 4. 이미지 열기 및 크기 측정
images = [Image.open(p) for p in image_paths]
img_width, img_height = images[0].size

# 5. 전체 그리드 캔버스 생성
grid_image = Image.new("RGB", (grid_cols * img_width, grid_rows * img_height), color="white")

# 6. 이미지 배치
for idx, img in enumerate(images):
    row = idx // grid_cols
    col = idx % grid_cols
    x = col * img_width
    y = row * img_height
    grid_image.paste(img, (x, y))

# 7. 저장
grid_image.save(output_path)
print(f"✅ Saved grid image as: {output_path}")