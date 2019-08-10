#pragma once

#include <string>
#include <tsl/robin_map.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace EmptySource {

	typedef char			NChar;
	typedef wchar_t			WChar;
	typedef std::string		NString;
	typedef std::wstring	WString;

	template<class T>
	using TArray = std::vector<T>;
	template<class T>
	using TArrayInitializer = std::initializer_list<T>;

	template<class K, class T>
	using TDictionary = tsl::robin_map<K, T>;

}

#ifdef ES_PLATFORM_CUDA
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
