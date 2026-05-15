# NeuroX: Neuromorphic Computing Co-Simulation Framework

NeuroX is a hardware-software co-simulation framework designed for Compute-in-Memory (CIM) and Processing-in-Memory (PIM) architectures. It enables **Hardware-Aware Training (HAT)** and **Architecture Search (NAS)** by seamlessly integrating physical hardware constraints into the PyTorch training loop.

## Core Philosophy: "Zero-Code Change"

NeuroX is built for algorithm engineers. You write standard PyTorch code; NeuroX handles the hardware.

1. **Define Model**: Write your model in pure PyTorch (e.g., `models/resnet.py`).
2. **Define Hardware**: Configure your chip architecture (Device -> Array -> Macro -> Chip).
3. **Transform**: Use `MappingTransformer` to automatically convert your model into a hardware-aware graph.
4. **Train & Estimate**: Train normally. The simulator handles quantization, noise, and PPA estimation under the hood.

## Project Structure

- **`models/`**: Standard PyTorch AI models (ResNet, BERT, etc.). **No NeuroX code here.**
- **`src/neurox/`**: The core simulator.
  - **`device/`**: Physical device models (RRAM, PCM).
  - **`circuit/`**: Peripheral circuits (ADC, DAC).
  - **`macro/`**: CIM Macros (Tiles).
  - **`chip/`**: Top-level chip architecture.
- **`examples/`**: End-to-end training and inference scripts.

## Quick Start

```bash
python examples/mnist/train.py
```

## Key Features

- **Automatic Graph Transformation**: Replaces `nn.Linear`/`nn.Conv2d` with `VirtualLayer` counterparts.
- **Physics-Based Modeling**: Simulates device non-idealities (noise, drift, retention).
- **PPA Estimation**: Detailed Power, Performance, and Area breakdown.
- **Autograd Support**: Fully differentiable hardware models using Straight-Through Estimator (STE).
