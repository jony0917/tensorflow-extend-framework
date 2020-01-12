

#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_


template<typename Device, typename T>
struct ExampleFunctor{
  void operator()(const Device& d, int size, const T* in, T* out);
};


#if GOOGLE_CUDA
//Partially specialize functor for GpuDevice
template<typename T>
struct ExampleFunctor<Eigen::GpuDevice, T>{
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};

#endif //GOOGLE_CUDA

#endif //KERNEL_EXAMPLE_H_