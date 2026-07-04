import torch, numpy as np, os, argparse


class qwen3_5_vl_vision(torch.nn.Module):
    def __init__(self, vlm, batch_size):
        super().__init__()
        self.merge_size = 2
        self.temporal_patch_size = 2
        self.patch_size = 16
        self.channel = 3
        self.vpm = vlm.visual
        self.batch_size = batch_size

    def forward(self, pixel_value, grid_thw):
        if self.batch_size == 1:
            patches = pixel_value.repeat(self.temporal_patch_size, 1, 1, 1)
        elif self.batch_size % self.temporal_patch_size == 1:
            patches = torch.cat((pixel_value[-1:].repeat(2, 1, 1, 1), pixel_value), dim=0)
        else:
            patches = pixel_value
        gt, gh, gw = grid_thw[0][0], grid_thw[0][1], grid_thw[0][2]
        patches = patches.reshape(gt, self.temporal_patch_size, self.channel,
                                  gh // self.merge_size, self.merge_size, self.patch_size,
                                  gw // self.merge_size, self.merge_size, self.patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        fp = patches.reshape(gt * gh * gw, self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)
        vo = self.vpm(fp, grid_thw)
        ie = vo.pooler_output
        ss = (grid_thw.prod(-1) // self.merge_size ** 2).tolist()
        return torch.split(ie, ss)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.path, torch_dtype=torch.float32,
                                      low_cpu_mem_usage=True, _attn_implementation="eager",
                                      trust_remote_code=True).eval().to(args.device)
    pv = torch.randn(1, 3, 448, 448, device=model.device, dtype=torch.float32)
    gt = torch.tensor([[1, 448 // 16, 448 // 16]], dtype=torch.int64)
    m = qwen3_5_vl_vision(model, 1)
    out = m(pv, gt)
    print("Output shape:", out[0].shape)
    os.makedirs("./onnx", exist_ok=True)
    torch.onnx.export(m, (pv, gt), "./onnx/qwen3.5_vision.onnx",
                      input_names=["pixel", "grid_thw"],
                      dynamic_axes={"pixel": {2: "h", 3: "w"}},
                      opset_version=15)
    print("Exported to ./onnx/qwen3.5_vision.onnx")
