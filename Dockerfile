FROM rust:stretch

# The image built from this dockerfile is used to run the OCR on any machine with docker.

# Install basic tools
RUN apt-get -y update \
    && apt-get -y install build-essential openssl libssl1.1 libssl-dev git make \
                        gcc pkg-config autoconf clang-7 llvm-7 llvm-7-dev \
    && ldconfig /usr/local/lib;

# Install pytorch dependencies
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip \
  && unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cpu.zip -d /usr/lib/;

ENV LIBTORCH=/usr/lib/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

WORKDIR /app
COPY . .