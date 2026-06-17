import os
import imageio.v2 as imageio
import os
import imageio


def save_as_gif(img_dir, out_name=None, base_name="", total_duration=2.0):
    if out_name is None:
        out_name = "animation"
    output_gif = img_dir.parent / f"{out_name}.gif"

    images = []

    for filename in sorted(os.listdir(img_dir)):
        if filename.startswith(base_name) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            path = img_dir / filename
            images.append(imageio.imread(path))

    if len(images) == 0:
        return

    frame_duration = total_duration / len(images)

    imageio.mimsave(
        output_gif,
        images,
        duration=frame_duration,
        loop=0,
        subrectangles=False,  # prevents frame merging
    )


def save_as_video(dir, base_name, total_duration=6.0, n=40):
    output_video = dir / "animation.mp4"

    images = []
    dir_img = dir / "img"

    for filename in sorted(os.listdir(dir_img)):
        if filename.startswith(base_name) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            images.append(imageio.imread(dir_img / filename))

    if len(images) == 0:
        return

    # enforce exactly n frames
    if len(images) >= n:
        images = images[-n:]
    else:
        last = images[-1]
        images = images + [last] * (n - len(images))

    fps = n / total_duration  # exact control of total duration

    writer = imageio.get_writer(
        output_video,
        fps=fps,
        codec="libx264",
    )

    for img in images:
        writer.append_data(img)

    writer.close()
