mkdir build                                                                                                                              
cd build
cmake \
  -DCMAKE_PREFIX_PATH=/home/mark/repos/spherical_code/opt/libtorch/share/cmake \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  ..
cmake --build . --config Release
cd ..