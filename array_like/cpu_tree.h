#include <stdio.h>
#include <vector>
#include "globals.h"

using namespace std;

void cpu_search_tree(vector<uint32_t> &keys, vector<uint32_t> &values, vector<uint32_t> &queries, vector<uint32_t> &answers, uint32_t num_nodes, uint32_t K, uint32_t idx);
