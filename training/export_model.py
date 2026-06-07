from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from common import ExperimentConfig, build_transforms, write_json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a PyTorch checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/champion.onnx"))
    parser.add_argument("--quantized-output", type=Path, default=Path("models/champion.quant.onnx"))
    parser.add_argument("--copy-pytorch", type=Path, default=Path("models/champion.pt"))
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    config = ExperimentConfig(**checkpoint["config"])
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, config.img_size, config.img_size)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    shutil.copy2(args.checkpoint, args.copy_pytorch)

    with torch.no_grad():
        torch_logits = model(dummy).numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"input": dummy.numpy()})[0]
    max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))

    quantized_ok = False
    quant_error = None
    try:
        quantize_dynamic(str(args.output), str(args.quantized_output), weight_type=QuantType.QInt8)
        quantized_ok = True
    except Exception as exc:
        quant_error = str(exc)

    metadata = {
        "model_name": checkpoint["model_name"],
        "class_names": class_names,
        "img_size": config.img_size,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "onnx_path": str(args.output.resolve()),
        "pytorch_copy": str(args.copy_pytorch.resolve()),
        "quantized_path": str(args.quantized_output.resolve()) if quantized_ok else None,
        "quantized_ok": quantized_ok,
        "quantization_error": quant_error,
        "torch_onnx_max_abs_diff": max_abs_diff,
        "checkpoint_metrics": checkpoint.get("metrics", {}),
    }
    write_json(args.output.parent / "model_metadata.json", metadata)
    write_json(args.output.parent / "classes.json", class_names)
    print(metadata)


if __name__ == "__main__":
    main()
