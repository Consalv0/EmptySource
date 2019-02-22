#pragma once

#include <vector>
template<class T>
using TArray = std::vector<T>;

#include <map>
template<class K, class T>
using TDictionary = std::map<K, T>;

#include <cuda_runtime.h>

#define HOST_DEVICE __host__ __device__ 
#define FORCEINLINE __forceinline	/* Force code to be inline */