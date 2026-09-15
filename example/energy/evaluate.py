"""Run checkpoint inference with independent NeuroX energy observation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from .factory import PRESETS, ROOT, UnitFactory
from .sidecar import EnergySidecar

DATA = {
    "ucihar": "data/ucihar/UCI_HAR_Dataset",
    "hhar": "data/hhar/Activity_recognition_exp",
    "uthar": "data/uthar",
    "aril": "data/aril",
    "fihumanid": "data/fihumanid",
    "bullydetect": "data/bullydetect",
    "widar": "data/widar",
    "urbansound": "data/urbansound",
    "gsc": "data/gsc",
    "cifar10": "data/cifar10",
    "cifar10dvs": "data/cifar10dvs",
    "dvsgesture": "data/dvsgesture",
}
FAMILIES = ("example", "hardware_comparable", "lenet_sparse_adc", "soul_fullgrid_sparse_adc")


def load_soul(args: argparse.Namespace) -> tuple[nn.Module, list, str]:
    root = ROOT / args.family
    sys.path.insert(0, str(root))
    os.environ["SOUL_ROOT"] = str(root)
    os.environ["SORBET_DIR"] = str(root)
    qfull = importlib.import_module("qfull")
    checkpoint = args.checkpoint or root / "checkpoints" / (
        f"{args.dataset}_s0.pt" if args.family == "lenet_sparse_adc" else f"{args.dataset}_{args.model}.pt"
    )
    data_dir = root / DATA[args.dataset]
    if not data_dir.exists():
        data_dir = ROOT / "soul_fullgrid_sparse_adc" / DATA[args.dataset]
    model, _, test, _ = qfull.build(args.dataset, str(data_dir), args.model, "lif", args.steps, 1)
    qfull.replace_linear(model, act_bits=8)
    qfull.replace_conv(model)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["state_dict"])
    layers = {name: (spike, span) for name, spike, span in checkpoint_data["layers"]}
    for name, layer in model.named_modules():
        if isinstance(layer, (qfull.QLinear, qfull.QConv)):
            layer.is_spike, layer.adc_range = layers[name]
            layer.sparse, layer.dense_mode = True, "bw"
            if isinstance(layer, qfull.QConv):
                layer.adc_uses_tile = True
    qfull.set_mode(model, mode="adc", adc_bits=args.reference_adc_bits, calib=False)
    selected = torch.randperm(len(test), generator=torch.Generator().manual_seed(args.seed))[: args.num_samples]
    samples = []
    for index in selected.tolist():
        x, label = test[index]
        samples.append(((x.unsqueeze(1).to(args.device),), {}, int(label)))
    return model.to(args.device).eval(), samples, str(checkpoint)


def load_example(args: argparse.Namespace) -> tuple[nn.Module, list, str]:
    if args.model == "lenet":
        from torchvision import datasets, transforms

        from example.lenet.model_float import LeNet5

        checkpoint = args.checkpoint or ROOT / "weight/lenet_float.pth"
        model = LeNet5()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        test = datasets.MNIST(
            ROOT / "dataset/mnist",
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        samples = [((test[i][0].unsqueeze(0).to(args.device),), {}, int(test[i][1])) for i in range(args.num_samples)]
    elif args.model == "bert":
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

        snapshot = next(
            (ROOT / "dataset/sst2/models--google--bert_uncased_L-4_H-512_A-8/snapshots").glob("*/config.json")
        ).parent
        config = BertConfig.from_pretrained(snapshot, num_labels=2, local_files_only=True)
        model = BertForSequenceClassification(config)
        checkpoint = args.checkpoint or ROOT / "weight/bert_small_float.pth"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        tokenizer = BertTokenizer.from_pretrained(snapshot, local_files_only=True)
        import csv

        with (ROOT / "hardware_comparable/data/SST-2/dev.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))[: args.num_samples]
        samples = [
            (
                (),
                {
                    k: v.to(args.device)
                    for k, v in tokenizer(
                        row["sentence"],
                        max_length=args.sequence_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).items()
                },
                int(row["label"]),
            )
            for row in rows
        ]
    elif args.model == "sorbet":
        return load_sorbet(args)
    else:
        raise ValueError(f"unsupported example model: {args.model}")
    return model.to(args.device).eval(), samples, str(checkpoint)


def load_sorbet(args: argparse.Namespace) -> tuple[nn.Module, list, str]:
    root = ROOT / ("hardware_comparable" if args.family == "hardware_comparable" else "example/Sorbet_SST2_infer")
    sys.path.insert(0, str(root))
    from transformer.configuration_bert import BertConfig
    from transformer.modeling_bert_quantize_icml26_mttfs import BertForSequenceClassification
    from transformer.tokenization import BertTokenizer
    from utils_glue import convert_examples_to_features, get_tensor_data, processors

    checkpoint = args.checkpoint or (
        root / "ckpt_6bit"
        if args.family == "hardware_comparable"
        else root / "outputs/SST-2/icml26-w1a4-k0-onemoreround/kd_joint"
    )
    config = BertConfig.from_pretrained(str(checkpoint))
    config.weight_bits, config.input_bits = 1, 4
    config.weight_quant_method, config.input_quant_method = "bwn", "elastic"
    config.clip_init_val, config.learnable_scaling = 2.5, True
    config.sym_quant_qkvo, config.sym_quant_ffn_attn = True, False
    config.embed_layerwise, config.weight_layerwise, config.input_layerwise = False, True, True
    config.hidden_act, config.not_quantize_attention = "relu", False
    model = BertForSequenceClassification.from_pretrained(str(checkpoint), config=config)
    tokenizer = BertTokenizer.from_pretrained(str(root / "models/SST-2"), do_lower_case=True)
    processor = processors["sst-2"]()
    data = root / ("data/SST-2" if args.family == "hardware_comparable" else "glue_data/SST-2")
    features = convert_examples_to_features(
        processor.get_dev_examples(str(data))[: args.num_samples],
        processor.get_labels(),
        args.sequence_length,
        tokenizer,
        "classification",
    )
    dataset, _ = get_tensor_data("classification", features)
    samples = [
        (
            (
                row[0].unsqueeze(0).to(args.device),
                row[2].unsqueeze(0).to(args.device),
                row[1].unsqueeze(0).to(args.device),
            ),
            {},
            int(row[3]),
        )
        for row in dataset
    ]
    return model.to(args.device).eval(), samples, str(checkpoint)


def logits(output: object) -> Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    return output[0]


def main(family: str | None = None, *, model: str = "lenet") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, default=family or "example")
    parser.add_argument("--model", default=model)
    parser.add_argument("--dataset", default="ucihar", choices=tuple(DATA))
    parser.add_argument("--preset", choices=PRESETS, default=PRESETS[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-adc-bits", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-chunk", type=int, default=1)
    parser.add_argument("--spatial-chunk", type=int, default=32)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    torch.manual_seed(args.seed)
    start = time.monotonic()
    loader = (
        load_example if args.family == "example" else load_sorbet if args.family == "hardware_comparable" else load_soul
    )
    model, samples, checkpoint = loader(args)
    sidecar = EnergySidecar(
        model,
        UnitFactory(args.preset, merge=args.merge),
        batch_chunk=args.batch_chunk,
        spatial_chunk=args.spatial_chunk,
    )
    correct = 0
    with torch.no_grad():
        for operands, kwargs, label in samples:
            state = {name: value.clone() for name, value in model.state_dict().items()}
            with torch.random.fork_rng(devices=[torch.device(args.device).index or 0] if "cuda" in args.device else []):
                expected = logits(model(*operands, **kwargs)).clone()
            model.load_state_dict(state)
            with sidecar:
                actual = logits(model(*operands, **kwargs))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            correct += int(actual.argmax(-1).item() == label)
    result = sidecar.report() | {
        "family": args.family,
        "model": args.model,
        "dataset": args.dataset
        if args.family.endswith("sparse_adc")
        else "mnist"
        if args.model.startswith("lenet")
        else "sst2",
        "checkpoint": checkpoint,
        "num_samples": len(samples),
        "correct": correct,
        "original_logits_unchanged": True,
        "wall_seconds": time.monotonic() - start,
    }
    if result["operator_calls"] == 0:
        raise RuntimeError("no supported operators were observed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(json.dumps({k: v for k, v in result.items() if k != "operators"}) + "\n")


if __name__ == "__main__":
    main()
