def get_highest_version(version_dirs):
    return max([int(x.split("_")[1]) for x in version_dirs])


def get_best_checkpoint(checkpoints, latest=True):
    return sorted(
        checkpoints,
        key=lambda c: (
            # First, sort by the score in the filename
            float(c.split("=")[-1].replace(".ckpt", "")),
            # Then, sort the epoch in descending order.
            # This means that any score ties will use the later epoch.
            (-1 if latest else 1) * int(c.split("=")[1].split("-")[0]),
        ),
    )[0]
