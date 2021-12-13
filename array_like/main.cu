#include <iostream>
#include <vector>
#include <ctime>
#include <cassert>
#include "globals.h"
#include "utils.cuh"
#include "kernels.cuh"
#include "cpu_tree.h"
using namespace std;



int main(void) {
  cout << "Starting program...\n";

  // number of values in dataset
  uint32_t VALUES = 1500*1000*1000;

  // width of b-tree
  uint32_t K = 10;

  // number of queries
  uint32_t BLOCKS = 10*1000;
  uint32_t THREADS = 1000;
  uint32_t NUM_Q = BLOCKS*THREADS;

  // number of nodes needed to build tree
  uint32_t num_nodes = get_num_nodes(VALUES, K);



  // host mem init
  std::vector<uint32_t> keys;
  std::vector<uint32_t> values;
  std::vector<uint32_t> queries;
  std::vector<uint32_t> answers;
  std::vector<uint32_t> cpu_answers;

  // memory for tree
  keys.reserve(num_nodes + VALUES);
  values.reserve(num_nodes + VALUES);

  // array of queries and answers
  queries.reserve(NUM_Q);
  answers.reserve(NUM_Q);
  cpu_answers.reserve(NUM_Q);

  // randomly select keys for queries
  srand(time(0));
  for (uint32_t i=0; i<NUM_Q; i++){
    queries.push_back(rand()%(VALUES-1));
  }


  // set keys in order and values randomly
  // - values simulate pointers to disc
  srand(time(0));
  for (uint32_t i=0; i<VALUES+num_nodes; i++){
    if (i>=num_nodes){
      keys.push_back(i-num_nodes);
      values.push_back(rand());
    } else{
      keys.push_back(0);
      values.push_back(0);
    }
  }

  cout << "query " << queries[0] << " answer " << values[num_nodes+queries[0]]<<endl;
  cout << "query " << queries[1] << " answer " << values[num_nodes+queries[1]]<<endl;

  // build non-leaf data structure
  get_upper_keys(keys, values, VALUES, K);

  // *** view tree ***
  // printtree(keys, K, VALUES);
  // printtree(values, K, VALUES);
  // // printv(keys);
  // printv(values);


  // device pointers
  uint32_t *d_keys, *d_values, *d_queries, *d_answers;
  VALUES = VALUES + num_nodes;

  // alloc and copy; and time it
  GpuTimer timer;
  timer.timerStart();
  GPU_CHK(cudaMalloc((void**)&d_keys, sizeof(uint32_t)*VALUES));
  GPU_CHK(cudaMalloc((void**)&d_values, sizeof(uint32_t)*VALUES));
  GPU_CHK(cudaMalloc((void**)&d_queries, sizeof(uint32_t)*NUM_Q));
  GPU_CHK(cudaMalloc((void**)&d_answers, sizeof(uint32_t)*NUM_Q));
  GPU_CHK(cudaMemcpy(d_keys, keys.data(), sizeof(uint32_t)*VALUES, cudaMemcpyHostToDevice));
  GPU_CHK(cudaMemcpy(d_values, values.data(), sizeof(uint32_t)*VALUES, cudaMemcpyHostToDevice));
  GPU_CHK(cudaMemcpy(d_queries, queries.data(), sizeof(uint32_t)*NUM_Q, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  timer.timerStop();
  cout << "time for device allocation memory transfer " << timer.getMsElapsed() << endl;

  // launch kernels; and time it
  timer.timerStart();
  search_tree<<<BLOCKS,THREADS>>>(d_keys, d_values, d_queries, d_answers, num_nodes, K);
  cudaDeviceSynchronize();
  timer.timerStop();
  cout << "time for " << NUM_Q << " queries "<< timer.getMsElapsed() << endl;

  // copy answers back to host
  GPU_CHK(cudaMemcpy(&answers[0], d_answers, sizeof(uint32_t)*NUM_Q, cudaMemcpyDeviceToHost));

  cout << "query " << queries[0] << " answer " << values[num_nodes+queries[0]]<<endl;
  cout << "answer " << answers[0] << endl;

  // CPU solution

  timer.timerStart();
  for (uint32_t i=0;i<NUM_Q;i++){
    cpu_search_tree(keys, values, queries, cpu_answers, num_nodes, K, i);
  }
  timer.timerStop();
  cout << "CPU || time for " << NUM_Q << " queries "<< timer.getMsElapsed() << endl;

  // check answers are correct
  for (uint32_t i=0;i<NUM_Q; i++){
    assert(answers[i] == values[num_nodes+queries[i]]);
  }
  cout << "GPU responses validated.\n";
  for (uint32_t i=0;i<NUM_Q; i++){
    assert(cpu_answers[i] == values[num_nodes+queries[i]]);
  }
  cout << "CPU responses validated.\n";

  return 0;
}
