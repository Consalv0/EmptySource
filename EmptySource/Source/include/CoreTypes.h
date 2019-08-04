#pragma once

#include <vector>
#include "../External/RobinMap/include/tsl/robin_map.h"

namespace EmptySource {

	template<class T>
	using TArray = std::vector<T>;

	template<class K, class T>
	using TDictionary = tsl::robin_map<K, T>;

}

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
