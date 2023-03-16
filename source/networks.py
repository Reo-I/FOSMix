import segmentation_models_pytorch as smp

def load_network(args):
    if args.network == "resnet50":
        net = smp.Unet(
            in_channels = 3,
            classes=args.n_class+1,
            activation=None,
            encoder_weights="imagenet",
            encoder_name="resnet50",
            decoder_attention_type="scse",
        )

    elif args.network == "resnet101":
        net = smp.Unet(
            in_channels = 3,
            classes=args.n_class+1,
            activation=None,
            encoder_weights="imagenet",
            encoder_name = "resnet101",
            decoder_attention_type="scse",
        )

    elif args.network == "vgg16":
        net = smp.Unet(
            in_channels = 3,
            classes=args.n_class+1,
            activation=None,
            encoder_weights="imagenet",
            encoder_name = "vgg16",
            decoder_attention_type="scse",
        )

    return net

def load_MFI_network(args):
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