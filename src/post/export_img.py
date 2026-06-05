import os
import imageio.v2 as imageio
import os
import imageio


def save_as_gif(dir, base_name, total_duration=2.0):
    output_gif = dir / "animation.gif"

    images = []
    dir_img = dir / "img"

    for filename in sorted(os.listdir(dir_img)):
        if filename.startswith(base_name) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            path = dir_img / filename
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
