# The image built from this dockerfile is used to run the OCR on any machine with docker.
FROM rust:buster

# Install tools
RUN apt-get -y update && apt-get -y install openssl clang-7 \
  # Install pytorch dependencies
  && wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.7.0.zip \
  && unzip libtorch-cxx11-abi-shared-with-deps-1.7.0.zip -d /usr/lib/ \
  # Cleanup unnecessary stuff
  && rm libtorch-cxx11-abi-shared-with-deps-1.7.0.zip;

ENV LIBTORCH=/usr/lib/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

WORKDIR /app
COPY . .