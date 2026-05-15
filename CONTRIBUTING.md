# Contributing to NeuroX

First off, thank you for considering contributing to NeuroX! It's people like you that make NeuroX such a great tool.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Documentation](#documentation)

## Getting Started

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS. Windows users are advised to use WSL2.
- **Python**: 3.8 or higher.
- **PyTorch**: 2.0 or higher.
- **Git**: For version control.

### Setting up the Development Environment

1. **Fork and Clone**
    Fork the repository on GitHub and clone your fork locally:

    ```bash
    git clone https://github.com/YOUR_USERNAME/NeuroX.git
    cd NeuroX
    ```

2. **Create a Virtual Environment**
    We recommend using `conda` or `venv`:

    ```bash
    conda create -n neurox python=3.9
    conda activate neurox
    ```

3. **Install Dependencies**
    Install the package in editable mode with development dependencies:

    ```bash
    pip install -r containers/env/requirements.txt
    pip install -r containers/env/requirements-dev.txt
    pip install -e .
    ```

4. **Verify Installation**
    Run the tests to ensure everything is set up correctly:

    ```bash
    pytest tests/
    ```

## Development Workflow

1. **Create a Branch**: Always work on a new branch for each feature or fix.

    ```bash
    git checkout -b feature/my-new-feature
    ```

2. **Commit Changes**: Make small, focused commits with clear messages.

    ```bash
    git commit -m "feat(device): add drift model for RRAM"
    ```

    *We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.*

3. **Keep it Synced**: Regularly pull upstream changes to avoid conflicts.

    ```bash
    git pull upstream master
    ```

4. **Format and Lint**: Before running tests or submitting, ensure code quality.

    ```bash
    ruff format .
    ruff check . --fix-only
    ```

5. **Submit a Pull Request (PR)**: Push your branch to your fork and open a PR against the `master` branch of the main repository.

## Coding Standards

To maintain high code quality and consistency, all contributors **must** adhere to the following standards.

### 1. Core Principles

- **Explicit is better than implicit**: Do not rely on global variables. Pass dependencies explicitly via `__init__`.
- **Pure Python/PyTorch**: Do not introduce C++ extensions. We rely on PyTorch's JIT or CUDA kernels if performance is critical, but logic must remain in Python.
- **Type Safety**: All function signatures must have Python Type Hints.
- **One Class Per File**: To avoid monolithic files, each major class should reside in its own file.
- **Clean Imports**: Use `__init__.py` files to expose public classes from subpackages.

### 2. Naming Conventions

- **Classes**: `PascalCase` (e.g., `SarAdc`, `RramDevice`)
- **Functions/Variables**: `snake_case` (e.g., `calculate_power`, `read_noise`)
- **Constants**: `UPPER_CASE` (e.g., `BOLTZMANN_CONSTANT`)
- **Private Members**: `_leading_underscore` (e.g., `_internal_state`)

### 3. Physical Quantity Naming (Strict)

To prevent physics errors and unit confusion, all variables, function arguments, and configuration keys representing physical quantities **MUST** include their unit as a suffix.

**Examples:**

- **Length**: `width_nm`, `length_um`, `distance_mm`
- **Time**: `delay_ns`, `period_ps`, `latency_cycles`
- **Voltage**: `supply_voltage_V`, `read_voltage_V`
- **Current**: `leakage_current_mA`, `drive_current_uA`
- **Power**: `static_power_mW`, `dynamic_power_mW`
- **Energy**: `dynamic_energy_pJ`, `energy_per_bit_pJ`
- **Capacitance**: `gate_capacitance_fF`, `load_capacitance_fF`
- **Resistance**: `wire_resistance_Ohm`, `on_resistance_kOhm`
- **Temperature**: `temperature_K`
- **Frequency**: `clock_frequency_GHz`
- **Memory Size**: `size_KB`, `size_MB` (Avoid `capacity` for memory size to distinguish from capacitance)

**Ambiguity Resolution:**

- Never use generic names like `leakage`, `power`, or `capacity`.
- **Bad**: `get_leakage()`, `self.width`, `config.capacity`
- **Good**: `get_leakage_power_mW()`, `self.width_nm`, `config.size_KB`

### 4. Tensor Shapes

- **Weight Matrices**: `[Out_Channels, In_Channels]` (Follows `torch.nn.Linear` convention).
- **Activations**: `[Batch_Size, Channels]` or `[Batch_Size, Channels, Height, Width]`.
- **Conductance Maps**: `[Batch_Size, Rows, Cols]`.

### 5. Error Handling

- Use `ValueError` for invalid arguments (e.g., negative resistance).
- Use `RuntimeError` for simulation state errors.
- **Never** use bare `assert` statements for control flow; use them only for internal invariant checking.

### 6. Configuration

- **No Hardcoding**: Do not hardcode parameters in Python files.
- **Config Files**: Use YAML/JSON files in a `configs/` directory for all tunable parameters.
- **Loading**: Use `neurox.utils.config.load_config` to instantiate configuration objects.
- **Usage**: Access parameters via `self.config`.

```python
# BAD
v_read = 0.1

# GOOD
v_read = self.config.read_voltage
```

## Project Structure

Before writing code, please familiarize yourself with the project architecture:

- **[Architecture Design](docs/design/architecture.md)**: Overview of the system, layers, and data flow.
- **[API Interfaces](docs/design/api_interfaces.md)**: Detailed specification of the `HardwareModule`, `forward`, and `estimate` interfaces.
- **[Reference Analysis](docs/design/reference_analysis.md)**: Background on why we made certain design choices based on Cross-Sim and NeuroSim.

### Adding New Examples

When adding a new example:

1. Create a new directory: `examples/<task_name>/`.
2. Add a `configs/` subdirectory with a default YAML configuration.
3. Place your `train.py` or `inference.py` in the task directory.

## Testing

We use `pytest` for testing.

- **Unit Tests**: Located in `tests/`. Mirror the structure of `src/neurox/`.
- **Coverage**: We aim for high test coverage. Please add tests for any new features.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_device.py
```

## Documentation

- **Docstrings**: All public classes and methods must have Google-style docstrings.
- **Type Hints**: Use `typing` module (e.g., `List`, `Dict`, `Optional`) or standard types.

```python
def calculate_energy(self, voltage: float, current: float, time: float) -> float:
    """
    Calculates the energy consumption.

    Args:
        voltage: Operating voltage in Volts.
        current: Operating current in Amperes.
        time: Duration in Seconds.

    Returns:
        Energy in Joules.
    """
    return voltage * current * time
```

---

Happy Coding!
