import torch
import torch.nn.functional as F

def grad_cam(model, x, target_class_idx: int = None, layer_attr: str = None):
    if layer_attr is None:
        layer_attr = next(reversed([n for n,_ in model.named_modules()]))
    acts = []; grads = []
    def fwd_hook(_, __, output): acts.append(output.detach())
    def bwd_hook(_, grad_in, grad_out): grads.append(grad_out[0].detach())
    layer = dict(model.named_modules())[layer_attr]
    hf = layer.register_forward_hook(fwd_hook)
    hb = layer.register_full_backward_hook(bwd_hook)
    model.zero_grad()
    logits = model(x)
    if target_class_idx is None:
        target_class_idx = logits.argmax(dim=1)
    one_hot = F.one_hot(target_class_idx, num_classes=logits.shape[1]).float()
    loss = (logits * one_hot).sum()
    loss.backward()
    A = acts[-1]; G = grads[-1]
    weights = G.mean(dim=(2,3), keepdim=True)
    cam = (weights * A).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam - cam.amin(dim=(2,3), keepdim=True)
    cam = cam / (cam.amax(dim=(2,3), keepdim=True) + 1e-9)
    hf.remove(); hb.remove()
    return cam
