
FROM tensorflow/tensorflow:2.1.0-gpu

RUN apt-get update && apt-get -y upgrade; \
    apt-get install -y build-essential cmake

RUN mkdir -p /usr/local/lib/python2.7/dist-packages/tensorflow_core/include/third_party/gpus/cuda && \
    ln -s /usr/local/cuda-10.1/targets/x86_64-linux/include/ /usr/local/lib/python2.7/dist-packages/tensorflow_core/include/third_party/gpus/cuda/include