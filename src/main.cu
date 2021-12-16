#include <iostream>
#include <vector>
#include <ctime>
#include <cassert>
#include "globals.h"
#include "utils.cuh"
#include "kernels.cuh"
#include "cpu_tree.h"
#include "math.h"
using namespace std;



int main(int argc, char** argv) {
  cout << BAR << "Starting program...\n";
  uint32_t VALUES, K, BLOCKS, THREADS, NUM_Q, managed;
  if (argc == 1){
    VALUES = 1000*1000*1000;
    K = 10;
    BLOCKS = 20*1000;
    THREADS = 1000;
    NUM_Q = BLOCKS*THREADS;
    managed = 0;
  } else {
    VALUES = 100*1000*1000;
    K = 10;
    NUM_Q = pow(10, atoi(argv[1]));
    BLOCKS = max(1,NUM_Q/1000);
    THREADS = NUM_Q/BLOCKS;
    managed = 0;
  }
  // } else if (argv[1][0] == '1'){
  //   VALUES = 100*1000*1000;
  //   K = 10;
  //   BLOCKS = 1000;
  //   THREADS = 1000;
  //   NUM_Q = BLOCKS*THREADS;
  //   managed = 0;
  // }  else if (argv[1][0] == '2'){
  //   VALUES = 100*1000*1000;
  //   K = 10;
  //   BLOCKS = 1;
  //   THREADS = 1000;
  //   NUM_Q = BLOCKS*THREADS;
  //   managed = 0;
  // }  else if (argv[1][0] == '3'){
  //   VALUES = 100*1000*1000;
  //   K = 10;
  //   BLOCKS = 1;
  //   THREADS = 100;
  //   NUM_Q = BLOCKS*THREADS;
  //   managed = 0;
  // }  else if (argv[1][0] == '4'){
  //   VALUES = 100*1000*1000;
  //   K = 10;
  //   BLOCKS = 1;
  //   THREADS = 1;
  //   NUM_Q = BLOCKS*THREADS;
  //   managed = 0;
  // } else {
  //   VALUES = 2100*1000*1000;
  //   K = 10;
  //   BLOCKS = 1*1000;
  //   THREADS = 1000;
  //   NUM_Q = BLOCKS*THREADS;
  //   managed = 1;
  // }
  cout << NUM_Q / 1000000 << " Million Queries\n";
  cout << float(VALUES) / float(1000000000) << " Billion Elements ~ " << float(VALUES) / float(1000000000)*8<< " Gigabytes\n\n";
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

  cout << "Randomly setting values...\n";
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


  cout << "building tree..\n\n";
  // build non-leaf data structure
  get_upper_keys(keys, values, VALUES, K);

  // *** view tree ***
  // printtree(keys, K, VALUES);
  // printtree(values, K, VALUES);
  // printv(keys);
  // printv(values);


  // device pointers
  uint32_t *d_keys, *d_values, *d_queries, *d_answers;
  VALUES = VALUES + num_nodes;

  // alloc and copy; and time it
  cout << "device alloc and transfer...\n";
  GpuTimer timer;
  timer.timerStart();
  if (!managed){
    GPU_CHK(cudaMalloc((void**)&d_keys, sizeof(uint32_t)*VALUES));
    GPU_CHK(cudaMalloc((void**)&d_values, sizeof(uint32_t)*VALUES));
    GPU_CHK(cudaMalloc((void**)&d_queries, sizeof(uint32_t)*NUM_Q));
    GPU_CHK(cudaMalloc((void**)&d_answers, sizeof(uint32_t)*NUM_Q));
    GPU_CHK(cudaMemcpy(d_keys, keys.data(), sizeof(uint32_t)*VALUES, cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(d_values, values.data(), sizeof(uint32_t)*VALUES, cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(d_queries, queries.data(), sizeof(uint32_t)*NUM_Q, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
  }
  else {
    GPU_CHK(cudaMallocManaged((void**)&d_keys, sizeof(uint32_t)*VALUES));
    GPU_CHK(cudaMallocManaged((void**)&d_values, sizeof(uint32_t)*VALUES));
    GPU_CHK(cudaMallocManaged((void**)&d_queries, sizeof(uint32_t)*NUM_Q));
    GPU_CHK(cudaMallocManaged((void**)&d_answers, sizeof(uint32_t)*NUM_Q));
    GPU_CHK(cudaMemcpy(d_keys, keys.data(), sizeof(uint32_t)*VALUES, cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(d_values, values.data(), sizeof(uint32_t)*VALUES, cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(d_queries, queries.data(), sizeof(uint32_t)*NUM_Q, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
  }
  timer.timerStop();
  cout << "time for device allocation memory transfer " << timer.getMsElapsed() << endl<<endl;

  // launch kernels; and time it
  timer.timerStart();
  cout << "GPU || Running Queries...\n";
  search_tree<<<BLOCKS,THREADS>>>(d_keys, d_values, d_queries, d_answers, num_nodes, K);
  cudaDeviceSynchronize();
  timer.timerStop();
  cout << "GPU || time for " << NUM_Q << " queries "<< timer.getMsElapsed() << endl << endl;

  // copy answers back to host
  GPU_CHK(cudaMemcpy(&answers[0], d_answers, sizeof(uint32_t)*NUM_Q, cudaMemcpyDeviceToHost));



  // CPU solution
cout << "CPU || Running Queries...\n";
  timer.timerStart();
  for (uint32_t i=0;i<NUM_Q;i++){
    cpu_search_tree(keys, values, queries, cpu_answers, num_nodes, K, i);
  }
  timer.timerStop();
  cout << "CPU || time for " << NUM_Q << " queries "<< timer.getMsElapsed() << endl << endl;

  // check answers are correct
  for (uint32_t i=0;i<NUM_Q; i++){
    assert(answers[i] == values[num_nodes+queries[i]]);
  }
  cout << "GPU responses validated.\n";
  for (uint32_t i=0;i<NUM_Q; i++){
    assert(cpu_answers[i] == values[num_nodes+queries[i]]);
  }
  cout << "CPU responses validated.\n" << BAR;

  return 0;
}
