#include "kernels.cuh"

__global__ void search_tree(uint32_t* keys, uint32_t* values, uint32_t* queries, uint32_t* answers, uint32_t num_nodes, uint32_t K){
  uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  uint32_t query_key = queries[idx];
  uint32_t answer;

  uint32_t iter = 0;
  while (1){
    for (uint32_t i = 0; i<K;i++){
      // printf("THREAD %u  || iter %u i %u value-ptr %u key % u\n", idx, iter, i, values[iter+i], keys[iter+i]);
      if (keys[iter+i] > keys[iter+1+i]){
        iter = values[iter+i];
        break;
      } else if (query_key < keys[iter+i+1]){
         if (iter < num_nodes){
           iter = values[iter+i];
         } else {
           iter += i;
         }
         break;
      }
    }
    if (iter > num_nodes and query_key == keys[iter]){
      break;
    }
  }
  answer = values[iter];
  answers[idx] = answer;
}
