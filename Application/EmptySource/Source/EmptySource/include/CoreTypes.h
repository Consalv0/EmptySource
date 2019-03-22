#pragma once

#include <vector>
template<class T>
using TArray = std::vector<T>;

#include "../External/tsl/robin_map.h"
template<class K, class T>
using TDictionary = tsl::robin_map<K, T>;

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#ifdef __APPLE__
#define FORCEINLINE inline
#else
#define FORCEINLINE __forceinline
#endif
