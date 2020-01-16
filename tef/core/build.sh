#/bin/dash

TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

echo ${TF_CFLAGS[@]}
echo ${TF_LFLAGS[@]}

/usr/local/cuda-10.1/bin/nvcc -std=c++11 -c -o kernels/example_op.cu.o kernels/example_op.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG
g++ -std=c++11 -shared ops/example_ops.cc kernels/zero_out_op.cc kernels/example_op.cc kernels/example_op.cu.o -o libtef_core.so -D GOOGLE_CUDA=1 -fPIC ${TF_CFLAGS[@]} -lcudart ${TF_LFLAGS[@]} -O2 -L/usr/local/cuda-10.1/targets/x86_64-linux/lib/
