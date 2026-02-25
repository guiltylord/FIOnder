from PIL import Image


def prepare_image(image: Image.Image) -> Image.Image:
    img = image.convert('L')
    
    scale = 4.0
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    
    return img
