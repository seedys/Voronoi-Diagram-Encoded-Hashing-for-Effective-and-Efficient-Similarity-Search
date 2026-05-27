# VDeH: Voronoi Diagram Encoded Hashing for Effective and Efficient Similarity Search

This repository contains the implementation code for the paper:

> **VDeH: Voronoi Diagram Encoded Hashing for Effective and Efficient Similarity Search, Applicable to Different Types of Databases**  
> Yang Xu, Kai Ming Ting, Xinpeng Li, Yunpeng Li  
> *Journal of Artificial Intelligence Research (JAIR)*, 2026

VDeH is a simple, efficient, and no-learning data-dependent binary hashing method based on Voronoi diagrams. It naturally possesses three key properties required for effective hashing — full space coverage, entropy maximization, and bit independence — without any optimization or learning process. VDeH serves as a versatile plug-and-play component for similarity search across diverse data domains, including vectors, graphs, and trajectories.

---

## Hardware Requirements

VDeH is developed using the AVX512 instruction set. Before running the code, please ensure your CPU supports the following flags: `avx512f`, `avx512cd`, `avx512vl`, `avx512dq`, `avx512bw`, `avx512_vpopcntdq`. For Linux operating systems, you can check using the following command:

```bash
lscpu | grep avx512
```

## Building from Source

VDeH is developed in C++ and can be built from source using CMake. The [test](./test/) folder contains test programs including [Exhaustive_Search_with_IKE.cpp](./test/Exhaustive_Search_with_IKE.cpp), [HNSW_with_IKE.cpp](./test/HNSW_with_IKE.cpp), and [IVF_with_IKE.cpp](./test/IVF_with_IKE.cpp). Please follow the steps below to compile:

### Step 1: Download the Source Code and Create a Build Directory

```bash
git clone https://github.com/Zed-Zed-b/VDeH.git
cd VDeH
mkdir build && cd build
```

### Step 2: Configure with CMake

Part of the implementation is based on the [Faiss library](https://github.com/facebookresearch/faiss) v1.9.0. Therefore, when running CMake, you need to specify the include path and the dynamic library path for Faiss. The simplest method is to create a new [conda](https://docs.conda.io/en/latest/) environment and install version 1.9.0 of the `faiss-cpu` package by following the instructions at [this document](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md). After activating this conda environment, run the following CMake command:

```bash
cmake .. -DFAISS_INCLUDE_ROOT=$CONDA_PREFIX/include -DFAISS_LIB_ROOT=$CONDA_PREFIX/lib
```

Alternatively, you can choose to compile and install the Faiss library manually from source and specify the corresponding paths using the `-DFAISS_INCLUDE_ROOT` and `-DFAISS_LIB_ROOT` options.

### Step 3: Compile with Make

```bash
make -j32
```

After completion, you will find the executable files in the `build` directory.

## Running the Programs

### Preparing the Dataset

We provide two pre-computed embedding datasets: [Istella22](https://istella.ai/datasets/istella22-dataset/) ([Qwen3-embedding](https://github.com/QwenLM/Qwen3-Embedding)) and [FEVER-HN](https://huggingface.co/datasets/mteb/FEVER_test_top_250_only_w_correct-v2) ([LLM2Vec](https://github.com/McGill-NLP/llm2vec)) [**here**](https://huggingface.co/datasets/Ang17/IKE_emb_data). Please download and extract the datasets (please ensure you are in the project's main directory):

```bash
mkdir data && cd data

# Download two datasets through shared URL

# Unzip
unzip feverHN_l2v.zip 
unzip istella22_qwen3.zip

# Remove zip files
rm feverHN_l2v.zip istella22_qwen3.zip

# Return to project root
cd ..
```

Now you can run the experiments! The following examples use the Istella22 (Qwen3-embedding) dataset.

**Note**: Since Istella22 is a multi-level relevance ranking retrieval dataset, you need to specify the path to the ground truth score file using the `-score` option to correctly calculate the retrieval metrics. The FEVER-HN dataset uses binary scores, so this option is not required.

### Exhaustive Search

```bash
./build/Exhaustive_Search_with_IKE -t 4096 --psi 11 --seed 0 -corpus "./data/istella22_qwen3/istella22_base_4096_qwen3.fvecs" -query "./data/istella22_qwen3/istella22_query_test_4096_qwen3.fvecs" -label "./data/istella22_qwen3/istella22_test_groundtruth.uivecs" -score "./data/istella22_qwen3/istella22_test_gt_scores.uivecs"
```

### HNSW Index

```bash
./build/HNSW_with_IKE -t 4096 --psi 11 --seed 0 -corpus "./data/istella22_qwen3/istella22_base_4096_qwen3.fvecs" -query "./data/istella22_qwen3/istella22_query_test_4096_qwen3.fvecs" -label "./data/istella22_qwen3/istella22_test_groundtruth.uivecs" -score "./data/istella22_qwen3/istella22_test_gt_scores.uivecs"
```

### IVF Index

```bash
./build/IVF_with_IKE -t 4096 --psi 11 --seed 0 -corpus "./data/istella22_qwen3/istella22_base_4096_qwen3.fvecs" -query "./data/istella22_qwen3/istella22_query_test_4096_qwen3.fvecs" -label "./data/istella22_qwen3/istella22_test_groundtruth.uivecs" -score "./data/istella22_qwen3/istella22_test_gt_scores.uivecs"
```

---

## Extension to NLP Retrieval

The VDeH method has been further extended and validated in the NLP domain for large-scale text retrieval. This extension work generalizes the isolation mechanism and integrates it with advanced industrial indices (HNSW, IVF), achieving significant improvements in throughput and memory efficiency on million-scale text retrieval benchmarks.

> **LLMs Meet Isolation Kernel: Lightweight, Learning-free Binary Embeddings for Fast Retrieval**  
> Zhibo Zhang, Yang Xu, Kai Ming Ting, Cam-Tu Nguyen  
> *ACL 2026*    

The code for this extension is available at: [https://github.com/Zed-Zed-b/IK_Emb](https://github.com/Zed-Zed-b/IK_Emb)

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{xu2026vdeh,
  title={VDeH: Voronoi Diagram Encoded Hashing for Effective and Efficient Similarity Search, Applicable to Different Types of Databases},
  author={Xu, Yang and Ting, Kai Ming and Li, Xinpeng and Li, Yunpeng},
  journal={Journal of Artificial Intelligence Research},
  year={2026}
}

@inproceedings{zhang2026ike,
  title={LLMs Meet Isolation Kernel: Lightweight, Learning-free Binary Embeddings for Fast Retrieval},
  author={Zhang, Zhibo and Xu, Yang and Ting, Kai Ming and Nguyen, Cam-Tu},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```
