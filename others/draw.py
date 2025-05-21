from PIL import Image, ImageDraw

def draw_window(draw, center, size, color):
    """
    Draws a square window on the image.

    :param draw: ImageDraw object
    :param center: Tuple of (x, y) coordinates
    :param size: Size of the square (e.g., 7 or 11)
    :param color: Color as a tuple (e.g., (255, 0, 0))
    """
    half = size // 2
    x, y = center
    top_left = (x - half, y - half)
    bottom_right = (x + half, y + half)
    draw.rectangle([top_left, bottom_right], outline=color, width=1)


    
def main(image_path, x, y, output_path="output.png"):
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
        normal_width, normal_height = image.size
        print (normal_width, normal_height)

    except Exception as e:
        print(f"Error loading image: {e}")
        return

    draw = ImageDraw.Draw(image)
    print (x, y)
    center = (x, y)

    # Draw 7x7 window (Green)
    draw_window(draw, center, 7, (0, 255, 0))

    # Draw 11x11 window (Red)
    draw_window(draw, center, 11, (255, 250, 250))
    draw_window(draw, center, 21, (255, 250, 250)) #10 per banda
    draw_window(draw, center, 31, (255, 250, 250)) #15 per banda

    # Save or display the image
    image.save(output_path)
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main("/data/uabcvmsc/shared/newborn/29/15.08.24/HM20240815060955.jpeg", 300, 300)
