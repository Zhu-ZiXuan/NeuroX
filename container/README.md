# NeuroX Container Environment

This directory contains the container configuration and Makefile targets for the NeuroX project.
We provide two main images:

- A lightweight **runtime** image `neurox:latest` for execution
- A feature-rich **development** image `neurox_dev:latest` for coding and testing.

All container operations are managed through `make` targets, usually invoked from the **project root**.

## Directory Structure

- **`Dockerfile`**: Defines the **runtime** image (`neurox`). Intended for running the application with only the runtime dependencies.
- **`Dockerfile.dev`**: Defines the **development** image (`neurox-dev`). Intended for development and testing.
- **`Makefile`**: Contains all targets for building and running the runtime/dev containers.
- **`README.md`**: This file.
- The primary (and only) dependency source is `pyproject.toml` in the project root.

## Prerequisites

- **Container engine**: Ensure a container engine (e.g. Docker, Podman) is running. We use `docker` by default for compatibility, and you can override it with `CONTAINER_ENGINE` variable.

```bash
make build_container CONTAINER_ENGINE=podman
make build_dev_container CONTAINER_ENGINE=podman
```

- **`pyproject.toml` at project root**: Dockerfiles use `pyproject.toml` at project root as the single source of dependencies. The build targets will fail when this file is missing.

## Variables

All variables below can be overridden on the `make` command line.

### Container engine

- `CONTAINER_ENGINE`: The container engine.
  - Default: `docker`.
  - Targets: `build_container`, `build_dev_container`, `run_container`, `run_dev_container`, `run_base_container`, `stop_dev_container`, `attach_dev_container`

### Images and names

- `CONTAINER_IMAGE`: The image name (target-specific).
  - Default for runtime: `neurox:latest`.
  - Default for dev: `neurox_dev:latest`.
  - Targets: `build_container`, `run_container`, `build_dev_container`, `run_dev_container`, `stop_dev_container`, `attach_dev_container`
- `CONTAINER_NAME`: The container name (target-specific).
  - Default for runtime: `neurox`.
  - Default for dev: `neurox_dev`.
  - Targets: `run_container`, `run_dev_container`, `stop_dev_container`, `attach_dev_container`
- `CONTAINER_BASE`: The base image:tag used to build NeuroX images (target-specific).
  - Default for runtime: `docker.io/pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime`.
  - Default for dev: `docker.io/pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime`.
  - Targets: `build_container`, `build_dev_container`.

### Runtime options

- `CONTAINER_GPU_OPT`: GPU-related options passed to the container engine.
  - Default: `--gpus=all` `--security-opt=label=disable`.
  - Targets: `run_container`, `run_dev_container`, `run_base_container`
- `CONTAINER_USER_OPT`: User-related options passed to the container engine.
  - Default: empty (use `root` in container).
  - Targets: `run_container`, `run_dev_container`, `run_base_container`
- `CONTAINER_PATH_OPT`: Path and working directory options passed to the container engine.
  - Default: `-v=$(PROJ_ROOT):$(PROJ_ROOT):z` `-w=$(PROJ_ROOT)`.
  - Targets: `run_container`, `run_dev_container`, `run_base_container`
- `CONTAINER_OPT`: Extra options passed to the container engine.
  - Default: empty, specify any other options you want to pass.
  - Targets: `run_container`, `run_dev_container`, `run_base_container`

## Building Images

Use `make` command in project root to manage container builds.

### Development Image

Build `neurox-dev:latest` with all development dependencies.

```bash
make build_dev_container
```

Or specify a custom base image using the `BASE` variable:

```bash
make build_dev_container BASE=docker.io/pytorch/pytorch:custom-tag
```

### Runtime Image

Build `neurox:latest` with only runtime dependencies.

```bash
make build_container
```

Or specify a custom base image using the `BASE` variable:

```bash
make build_container BASE=docker.io/pytorch/pytorch:custom-tag
```

## Running Containers

Use `make` command in project root to run containers with the correct volume mounts and GPU configurations.

### Development Container

Manage develop container.

```bash
make run_dev_container
make attach_dev_container
make stop_dev_container
```

Or customize the container name using the `NAME` variable:

```bash
make run_dev_container NAME=my_dev_container
make attach_dev_container NAME=my_dev_container
make stop_dev_container NAME=my_dev_container
```

### Runtime Container

Manage runtime container.

```bash
make run_container
```

Or customize the container name using the `NAME` variable:

```bash
make run_container NAME=my-runtime_container
```
