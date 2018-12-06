#pragma once

#include <vector>
template<class T>
using TArray = std::vector<T>;

#include <map>
template<class K, class T>
using TDictionary = std::map<K, T>;

#define FORCEINLINE __forceinline	/* Force code to be inline */