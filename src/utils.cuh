// #pragma once
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <algorithm>
#include "math.h"

#define GPU_CHK(call) { gpuAssert((call), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
//   do {
//     cudaError_t err = call;
//     if (err != cudaSuccess) {
//       printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
//       exit(EXIT_FAILURE);
//     }
// } while (0)


void get_upper_keys(std::vector<uint32_t> &keys, std::vector<uint32_t> &values, uint32_t len, uint32_t K);
void get_upper_ptrs(std::vector<uint32_t> &keys, std::vector<uint32_t> &values, uint32_t len, uint32_t K);
uint32_t get_num_nodes(uint32_t len, uint32_t K);
void printv(std::vector<uint32_t> &v);
void printtree(std::vector<uint32_t> &v, uint32_t K, uint32_t len);


class GpuTimer {
 public:
  GpuTimer() {}
  void timerStart() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
  }
  void timerStop() {
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  float getMsElapsed() { return temp_time; }

  float getSElapsed() { return temp_time * 0.001f; }
  ~GpuTimer(){};

 private:
  float temp_time = 0.0f;
  cudaEvent_t start, stop;
};


#endif
