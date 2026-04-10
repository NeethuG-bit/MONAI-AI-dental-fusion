from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensityRange

def get_transforms():
    pan_transform = Compose([
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensityRange(-1000, 1000, 0.0, 1.0, clip=True),
    ])

    cbct_transform = Compose([
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensityRange(-1000, 3000, 0.0, 1.0, clip=True),
    ])

    soft_transform = Compose([
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensityRange(0, 15, 0.0, 1.0, clip=True),
    ])

    return pan_transform, cbct_transform, soft_transform