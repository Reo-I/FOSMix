"""
Load the main segmentation model and the mask generator.
"""

import segmentation_models_pytorch as smp

def load_network(args, n_classes:int):
    """
    Loads a U-Net with the specified number of classes
    the type of which is determined by the 'network' argument.

    Args:
        args: Contains the network type
        n_classes (int): The number of classes to be used in the U-Net.

    Returns:
        net: The segmentation model i.e., U-Net model.
    """
    if args.network == "resnet50":
        net = smp.Unet(
            in_channels = 3,
            classes=n_classes,
            activation=None,
            encoder_weights="imagenet",
            encoder_name="resnet50",
            decoder_attention_type="scse",
        )
    elif args.network == "resnet50_ibn_a":
        net = smp.Unet(
            in_channels = 3,
            classes=n_classes,
            activation=None,
            encoder_weights="imagenet",
            encoder_name = "resnet50_ibn_a",
            decoder_attention_type="scse",
        )
    elif args.network == "resnet50_ibn_b":
        net = smp.Unet(
            in_channels = 3,
            classes=n_classes,
            activation=None,
            encoder_weights="imagenet",
            encoder_name = "resnet50_ibn_b",
            decoder_attention_type="scse",
        )
    elif args.network == "resnet101":
        net = smp.Unet(
            in_channels = 3,
            classes=n_classes,
            activation=None,
            encoder_weights="imagenet",
            encoder_name = "resnet101",
            decoder_attention_type="scse",
        )

    elif args.network == "vgg16":
        net = smp.Unet(
            in_channels = 3,
            classes=n_classes,
            activation=None,
            encoder_weights="imagenet",
            encoder_name = "vgg16",
            decoder_attention_type="scse",
        )

    return net

def load_MFI_network(args):
    """
    Load the mask generator with a specific number of input channels, 
    determined by whether reference images are used.
    
    If using source domain images only, the input channel count is 3. 
    If using reference images as well, the count is 6.

    """
    if args.G_use_ref == 0:
        input_channels = 3
    else:
        input_channels = 6

    if args.network == "vgg16":
        net = smp.Unet(
            in_channels = input_channels,
            classes=3,
            activation="sigmoid",
            encoder_weights="imagenet",
            encoder_name = "vgg11",
            decoder_attention_type="scse",
        )
    elif "resnet" in args.network:
        net = smp.Unet(
            in_channels = input_channels,
            classes=3,
            activation="sigmoid",
            encoder_weights="imagenet",
            encoder_name = "resnet18",
            decoder_attention_type="scse",
        )

    return net