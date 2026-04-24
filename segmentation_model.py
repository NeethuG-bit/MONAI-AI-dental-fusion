import torch
from monai.networks.nets import UNet


def build_segmentation_model(device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=2,
    ).to(device)

    return model


def run_segmentation(cbct_tensor, device, weights_path="segmentation_model.pth"):
    model = build_segmentation_model(device)

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        with torch.no_grad():
            pred = model(cbct_tensor)
            mask = torch.sigmoid(pred)
            mask = (mask > 0.5).float()

        return mask, "real"

    except Exception:
        mask = (cbct_tensor > cbct_tensor.mean()).float()
        return mask, "fallback"