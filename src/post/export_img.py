import os
import imageio.v2 as imageio


def save_as_gif(dir, base_name):
    # Folder containing your images
    output_gif = dir / f"animation.gif"

    # Collect all image files (sorted)
    images = []
    for filename in sorted(os.listdir(dir)):
        if filename.startswith(base_name) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            image_path = os.path.join(dir, filename)
            images.append(imageio.imread(image_path))

    # Save as GIF
    if len(images) > 0:
        imageio.mimsave(
            output_gif, images, duration=2.0
        )  # duration = time per frame in seconds
