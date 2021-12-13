#include "utils.cuh"
void get_upper_keys(std::vector<uint32_t> &keys, std::vector<uint32_t> &values, uint32_t len, uint32_t K){
  uint32_t num_nodes = get_num_nodes(len, K);
  uint32_t len2 = len;
  std::vector<uint32_t> li;
  li.push_back(len2);
  while (len2>K){
    len2 = len2/K + (len2 % K != 0);
    li.push_back(len2);
  }
  std::reverse(li.begin(), li.end());
  uint32_t idx = 0;
  for (uint32_t x=0;x<li.size()-1;x++){
    for (uint32_t i=0;i<li[x];i++){
      uint32_t level = li.size()-x-1;
      keys[idx] = i*pow(K,level);//i*K*len/li[x+1];
      values[idx] = idx+li[x]-i+i*K;
      idx+=1;
    }
  }
}
void get_upper_ptrs(std::vector<uint32_t> &keys, std::vector<uint32_t> &values, uint32_t len, uint32_t K){
  uint32_t num_nodes = get_num_nodes(len, K);
  std::vector<uint32_t> upper_ptrs;
  upper_ptrs.reserve(num_nodes);

}

uint32_t get_num_nodes(uint32_t len, uint32_t K){
  uint32_t count = 0;
  while (len>K){
    len = len/K + (len % K != 0);
    count += len;
  }
  return count;
}

void printv(std::vector<uint32_t> &v){
  std::cout<<"vec size " << v.size()<<std::endl;
  for (uint32_t x: v){
    std::cout<< x<< ' ';
  }
  std::cout<<std::endl;
}

void printtree(std::vector<uint32_t> &v, uint32_t K, uint32_t len){
  std::cout<<"vec size " << v.size()<<std::endl;

  std::vector<uint32_t> li;
  li.push_back(len);
  while (len>K){
    len = len/K + (len % K != 0);
    li.push_back(len);
  }
  std::reverse(li.begin(), li.end());

  uint32_t idx = 0;
  for (uint32_t x: li){
    for(uint32_t i=0;i<x;i++){
      std::cout<<v[idx]<< ' ';
      idx+=1;
    }
    std::cout<<std::endl;
  }

}
