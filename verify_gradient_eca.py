"""Offline checks for gradient ECA; does not run training or touch output/."""
from unittest.mock import patch

import torch

import model


def main():
    torch.manual_seed(55)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = model.ECABlock(16, use_spatial_gradient=False).to(device)
    fusion = model.ECABlock(16, use_spatial_gradient=True).to(device)
    missing, unexpected = fusion.load_state_dict(baseline.state_dict(), strict=False)
    assert missing == ["grad_conv.weight"] and not unexpected

    x = torch.randn(2, 16, 17, 19, device=device, requires_grad=True)
    torch.testing.assert_close(fusion(x), baseline(x), rtol=0, atol=0)
    print("PASS: zero-initialised fusion exactly matches original ECA")

    constant = torch.ones(1, 16, 9, 11, device=device)
    torch.testing.assert_close(
        fusion.spatial_gradient_descriptor(constant), torch.zeros_like(constant[:, :, :1, :1])
    )
    ramp = torch.arange(11., device=device).view(1, 1, 1, 11).expand(1, 16, 9, 11)
    assert (fusion.spatial_gradient_descriptor(ramp) > 0).all()
    torch.testing.assert_close(
        fusion.spatial_gradient_descriptor(ramp),
        fusion.spatial_gradient_descriptor(ramp * 10),
    )
    for height, width in [(1, 1), (1, 7), (7, 1)]:
        result = fusion(torch.randn(1, 16, height, width, device=device))
        assert torch.isfinite(result).all()
    print("PASS: constant/edge maps, amplitude scaling, and small spatial sizes")

    optimizer = torch.optim.SGD(fusion.parameters(), lr=0.01)
    before_step = fusion(x).detach().clone()
    fusion(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grad = fusion.grad_conv.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    optimizer.step()
    assert fusion.grad_conv.weight.detach().abs().sum() > 0
    assert not torch.equal(fusion(x).detach(), before_step)
    # Isolate the learned gradient contribution from changes in avg parameters.
    baseline.load_state_dict({
        name: value for name, value in fusion.state_dict().items()
        if name != "grad_conv.weight"
    })
    assert not torch.equal(fusion(x).detach(), baseline(x).detach())
    print("PASS: new branch receives gradients, updates, and affects attention")

    # Exercise the real model without downloading encoder weights.
    original_create_model = model.create_model

    def create_without_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    with patch.object(model, "create_model", side_effect=create_without_pretrained):
        network = model.build_model().to(device).eval()
    image = torch.randn(1, 3, model.H, model.W, device=device)
    main_output, aux_outputs, edge_outputs = network(image)
    outputs = [main_output, *aux_outputs, *edge_outputs]
    assert len(aux_outputs) == 3 and len(edge_outputs) == 3
    for output in outputs:
        assert output.shape[-2:] == image.shape[-2:]
        assert torch.isfinite(output).all()
    sum(output.square().mean() for output in outputs).backward()
    blocks = [module for module in network.modules() if isinstance(module, model.ECABlock)]
    for block in blocks:
        assert block.use_spatial_gradient
        assert block.grad_conv.weight.grad is not None
        assert torch.isfinite(block.grad_conv.weight.grad).all()
    print(f"PASS: full model forward/backward, {len(blocks)} gradient ECA blocks on {device}")


if __name__ == "__main__":
    main()
