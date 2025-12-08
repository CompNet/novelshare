FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD . /novelties-bookshare
WORKDIR /novelties-bookshare

RUN apt update -y

RUN apt install -y git
RUN uv sync --extra transformers-cuda

CMD ["bash"]