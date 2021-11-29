#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <windows.h>
#include <time.h>

using namespace std;


class TreeNode {
    int* keys;
    int t;
    TreeNode** C;
    int n;
    bool leaf;

public:
    TreeNode(int temp, bool bool_leaf);

    void insertNonFull(int k);
    void splitChild(int i, TreeNode* y);
    void traverse();

    TreeNode* search(int k);

    friend class BTree;
};

class BTree {
    TreeNode* root;
    int t;

public:
    BTree(int temp) {
        root = NULL;
        t = temp;
    }

    void traverse() {
        if (root != NULL)
            root->traverse();
    }

    TreeNode* search(int k) {
        return (root == NULL) ? NULL : root->search(k);
    }

    void insert(int k);
};

TreeNode::TreeNode(int t1, bool leaf1) {
    t = t1;
    leaf = leaf1;

    keys = new int[2 * t - 1];
    C = new TreeNode * [2 * t];

    n = 0;
}

void TreeNode::traverse() {
    int i;
    for (i = 0; i < n; i++) {
        if (leaf == false)
            C[i]->traverse();
        cout << " " << keys[i];
    }

    if (leaf == false)
        C[i]->traverse();
}

TreeNode* TreeNode::search(int k) {
    int i = 0;
    while (i < n && k > keys[i])
        i++;

    if (keys[i] == k)
        return this;

    if (leaf == true)
        return NULL;

    return C[i]->search(k);
}

void BTree::insert(int k) {
    if (root == NULL) {
        root = new TreeNode(t, true);
        root->keys[0] = k;
        root->n = 1;
    }
    else {
        if (root->n == 2 * t - 1) {
            TreeNode* s = new TreeNode(t, false);

            s->C[0] = root;

            s->splitChild(0, root);

            int i = 0;
            if (s->keys[0] < k)
                i++;
            s->C[i]->insertNonFull(k);

            root = s;
        }
        else
            root->insertNonFull(k);
    }
}

void TreeNode::insertNonFull(int k) {
    int i = n - 1;

    if (leaf == true) {
        while (i >= 0 && keys[i] > k) {
            keys[i + 1] = keys[i];
            i--;
        }

        keys[i + 1] = k;
        n = n + 1;
    }
    else {
        while (i >= 0 && keys[i] > k)
            i--;

        if (C[i + 1]->n == 2 * t - 1) {
            splitChild(i + 1, C[i + 1]);

            if (keys[i + 1] < k)
                i++;
        }
        C[i + 1]->insertNonFull(k);
    }
}

void TreeNode::splitChild(int i, TreeNode* y) {
    TreeNode* z = new TreeNode(y->t, y->leaf);
    z->n = t - 1;

    for (int j = 0; j < t - 1; j++)
        z->keys[j] = y->keys[j + t];

    if (y->leaf == false) {
        for (int j = 0; j < t; j++)
            z->C[j] = y->C[j + t];
    }

    y->n = t - 1;
    for (int j = n; j >= i + 1; j--)
        C[j + 1] = C[j];

    C[i + 1] = z;

    for (int j = n - 1; j >= i; j--)
        keys[j + 1] = keys[j];

    keys[i] = y->keys[t - 1];
    n = n + 1;
}

__global__ void gpu_search(BTree t, int* k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
}


int main() {
    BTree t(10);
    int n = 100000;
    int* d_k;

    for (int i = 0; i < n; i++) {
        t.insert(i*3);
    }

    cout << "The B-tree is: ";
    t.traverse();

    long starttime = clock();
    int k = 10;
    (t.search(k) != NULL) ? cout << endl
        << k << " was found"
        : cout << endl
        << k << " was not found";
    long endtime = clock();
    cout << " in " << (endtime - starttime) << " millis\n" << endl;

    long starttime2 = clock();
    k = 3;
    (t.search(k) != NULL) ? cout << endl
        << k << " was found"
        : cout << endl
        << k << " was not found";
    long endtime2 = clock();
    cout << " in " << (endtime2 - starttime2) << " millis\n" << endl;


    long starttime3 = clock();
    BTree* pT;
    cudaMallocManaged(&pT, sizeof(BTree));
    cudaMalloc(&d_k, sizeof(int));
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    *pT = t;
    gpu_search << <n / 320, 320 >> > (pT, k);
    long endtime3 = clock();
    cout << " CUDA run in " << (endtime2 - starttime2) << " millis\n" << endl;

    cudaDeviceSynchronize();
    cudaFree(pT);
}