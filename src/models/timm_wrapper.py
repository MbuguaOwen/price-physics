import timm, torch, torch.nn as nn

def patch_first_conv(model: torch.nn.Module, in_ch: int):
    first = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            first = m; break
    if first is None or first.in_channels == in_ch:
        return model
    new_conv = nn.Conv2d(in_ch, first.out_channels, kernel_size=first.kernel_size,
                         stride=first.stride, padding=first.padding, bias=first.bias is not None)
    with torch.no_grad():
        w = first.weight
        rep = in_ch // 3
        remainder = in_ch % 3
        w_rep = w.repeat(1, rep + (1 if remainder else 0), 1, 1)[:, :in_ch, :, :]
        new_conv.weight.copy_(w_rep / (in_ch/3.0))
        if new_conv.bias is not None: new_conv.bias.zero_()
    for name, module in model.named_modules():
        for cname, child in module.named_children():
            if child is first:
                setattr(module, cname, new_conv); return model
    return model

def build_timm(model_name: str, num_classes: int, in_ch: int):
    if model_name == "timm_convnext_tiny":
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes, in_chans=3)
    elif model_name == "timm_vit_tiny":
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes, in_chans=3)
    else:
        raise ValueError("Unknown timm model")
    return patch_first_conv(model, in_ch)
