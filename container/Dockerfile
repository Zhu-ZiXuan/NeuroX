# Copyright (c) 2025 ZHU ZiXuan
# SPDX-License-Identifier: MIT


ARG BASE_IMAGE


# === runtime image ===

FROM ${BASE_IMAGE} AS runtime

RUN apt-get update && apt-get install -y \
    make git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_BREAK_SYSTEM_PACKAGES=1

COPY pyproject.toml .

# pytorch base image provide `uv`, use it
# extract dependencies from `pyproject.toml` as the single source of truth
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --quiet --no-cache --system -r pyproject.toml \
    && rm -f pyproject.toml

CMD ["/bin/bash"]



# === devel image ===

FROM ${BASE_IMAGE} AS devel

RUN apt-get update && apt-get install -y \
    make git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_BREAK_SYSTEM_PACKAGES=1

COPY pyproject.toml .

# pytorch base image provide `uv`, use it
# extract dependencies from `pyproject.toml` as the single source of truth
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --quiet --no-cache --system -r pyproject.toml \
    --extra dev --extra doc \
    && uv pip install --quiet --no-cache --system nvitop \
    && rm -f pyproject.toml

CMD ["/bin/bash"]
