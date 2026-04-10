import matplotlib.pyplot as plt
import os

def _as_2d_for_imshow(arr):
    """2D images are shown as-is; 3D volumes use the middle slice along the last axis."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        z = arr.shape[-1] // 2
        return arr[:, :, z]
    raise ValueError(f"Expected 2D or 3D array after batch/channel squeeze, got shape {arr.shape}")


def visualize(pan, cbct, soft, output):
    pan = pan.cpu().numpy()[0, 0]
    cbct = cbct.cpu().numpy()[0, 0]
    soft = soft.cpu().numpy()[0, 0]
    output = output.detach().cpu().numpy()[0, 0]

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    axs[0].imshow(_as_2d_for_imshow(pan), cmap="viridis")
    axs[0].set_title("Panoramic")

    axs[1].imshow(_as_2d_for_imshow(cbct), cmap="viridis")
    axs[1].set_title("CBCT")

    axs[2].imshow(_as_2d_for_imshow(soft), cmap="viridis")
    axs[2].set_title("Soft Tissue")

    axs[3].imshow(_as_2d_for_imshow(output), cmap="viridis")
    axs[3].set_title("Output")

    for ax in axs:
        ax.axis("off")

    # ✅ SAVE IMAGE
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/result.png")

    plt.show()