import os
import random
import string
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

# ===============================
# 🔧 CONFIGURATION
# ===============================

FONT_DIR = "C:/Windows/Fonts"          # Path to system fonts (adjust as needed for Linux/macOS)
OUTPUT_DIR = "synthetic_captchas"      # Directory to save generated images
IMAGE_SIZE = (160, 60)                 # CAPTCHA image dimensions (width x height)
CHARS = string.ascii_uppercase + string.digits  # Characters used: A–Z + 0–9
CHARS_PER_IMAGE = 6                   # Number of characters per CAPTCHA
FONT_NAME = "arial.ttf"               # Font to render characters (must exist in FONT_DIR)
FONT_SIZE = 24                        # Size of the characters inside the bubbles
TOTAL_IMAGES = 10000                  # Total number of CAPTCHA images to generate

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# 🖨️ PRINT BANNER FOR CONSOLE
# ===============================
def print_banner():
    """Prints a stylized banner showing configuration."""
    print("=" * 60)
    print("🧠 SYNTHETIC CAPTCHA IMAGE GENERATOR with Python & PIL")
    print("🚀 Generating", TOTAL_IMAGES, "images using font:", FONT_NAME)
    print("📦 Output directory:", OUTPUT_DIR)
    print("=" * 60)


# ===============================
# 🔠 LOAD CUSTOM FONT
# ===============================
def get_fixed_font(size=FONT_SIZE):
    """Loads a fixed TrueType font from FONT_DIR."""
    font_path = os.path.join(FONT_DIR, FONT_NAME)
    if not os.path.exists(font_path):
        raise RuntimeError(f"❌ Font {FONT_NAME} not found in {FONT_DIR}.")
    return ImageFont.truetype(font_path, size)


# ===============================
# 🎨 DRAW RANDOM BACKGROUND CIRCLES
# ===============================
def add_color_circles(draw, width, height, count=30):
    """Draws randomized colored circles as background noise."""
    for _ in range(count):
        x = random.randint(-30, width)
        y = random.randint(-30, height)
        r = random.randint(5, 15)  # Random radius
        fill = tuple(random.randint(50, 255) for _ in range(3))  # Bright fill
        draw.ellipse((x, y, x + r, y + r), fill=fill)


# ===============================
# 🧪 GENERATE CAPTCHA IMAGE
# ===============================
def generate_captcha(text):
    """
    Generates a single CAPTCHA image with given text.
    Each character is placed inside a colored ellipse ("bubble"),
    rotated, and pasted onto the background.
    """
    image = Image.new('RGB', IMAGE_SIZE, (255, 255, 255))  # White canvas
    draw = ImageDraw.Draw(image)

    # Add random background noise
    add_color_circles(draw, *IMAGE_SIZE)

    spacing = IMAGE_SIZE[0] // (CHARS_PER_IMAGE + 1)  # Horizontal spacing between characters
    char_y = 10  # Vertical alignment offset
    font = get_fixed_font()  # Load consistent font

    for i, char in enumerate(text):
        # Create a transparent image for each bubble-character
        char_img = Image.new('RGBA', (40, 40), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)

        # Draw a colored circle (the "bubble")
        bubble_color = tuple(random.randint(0, 150) for _ in range(3))  # Dark-ish bubble
        char_draw.ellipse((4, 4, 36, 36), fill=bubble_color)

        # Center the character inside the bubble
        bbox = font.getbbox(char)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = (32 - w) // 2 - bbox[0]  # Center horizontally
        text_y = (32 - h) // 2 - bbox[1]  # Center vertically
        char_draw.text((text_x, text_y), char, font=font, fill=(255, 255, 255, 255))

        # Rotate the character image
        rotated_char = char_img.rotate(random.uniform(-25, 25), resample=Image.BICUBIC, expand=True)

        # Calculate final position and paste it onto the main image
        x_offset = spacing * (i + 1) - 20 + random.randint(-5, 5)
        y_offset = char_y + random.randint(-3, 3)
        image.paste(rotated_char, (x_offset, y_offset), rotated_char)

    # Slight smoothing to reduce sharpness and return final RGB image
    return image.filter(ImageFilter.SMOOTH_MORE).convert("RGB")


# ===============================
# 🧪 MAIN EXECUTION
# ===============================
print_banner()
start_time = time.time()

# Progress bar loop using tqdm
for i in tqdm(range(TOTAL_IMAGES), desc="🔧 Generating", unit="img"):
    label = ''.join(random.choices(CHARS, k=CHARS_PER_IMAGE))  # Generate random label like "A8K9ZB"
    img = generate_captcha(label)
    filename = f"{label}.jpeg"
    img.save(os.path.join(OUTPUT_DIR, filename))

    # Save first image as preview
    if i == 0:
        img.save("preview.jpeg")

    tqdm.write(f"📸 Saved: {filename}")  # Show current file saved

# Finish timing
total_time = time.time() - start_time
avg_speed = TOTAL_IMAGES / total_time

# Final summary output
print("\n✅ DONE!")
print(f"⏱️ Total time: {total_time:.2f} seconds")
print(f"⚡ Avg speed: {avg_speed:.2f} images/sec")
print(f"📂 Preview image saved as: preview.jpeg")
