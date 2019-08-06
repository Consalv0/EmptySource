#pragma once

#include "Engine/CoreTypes.h"

namespace EmptySource {

	inline void HashCombine(std::size_t& seed) { }

	template <typename T, typename... Rest>
	inline void HashCombine(std::size_t& seed, const T& v, Rest... rest) {
		std::hash<T> Hasher;
		seed ^= Hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		HashCombine(seed, rest...);
	}


	inline size_t WStringToHash(const WString & Name) {
		static const std::hash<WString> Hasher;
		return Hasher(Name);
	}

	// Code taken from https://stackoverflow.com/a/28801005 by tux3
	// Generate CRC lookup table
	template <unsigned C, int K = 8>
	struct GenCRCTable : GenCRCTable<((C & 1) ? 0xedb88320 : 0) ^ (C >> 1), K - 1> {};
	template <unsigned C> struct GenCRCTable<C, 0> { enum { Value = C }; };

#define A(x) B(x) B(x + 128)
#define B(x) C(x) C(x +  64)
#define C(x) D(x) D(x +  32)
#define D(x) E(x) E(x +  16)
#define E(x) F(x) F(x +   8)
#define F(x) G(x) G(x +   4)
#define G(x) H(x) H(x +   2)
#define H(x) I(x) I(x +   1)
#define I(x) GenCRCTable<x>::Value,

	constexpr unsigned CRCTable[] = { A(0) };

	// Constexpr implementation and helpers
	constexpr uint32_t CRC32Implementation(const uint8_t* p, size_t len, uint32_t crc) {
		return len ?
			CRC32Implementation(p + 1, len - 1, (crc >> 8) ^ CRCTable[(crc & 0xFF) ^ *p])
			: crc;
	}

	constexpr uint32_t EncodeCRC32(const uint8_t* data, size_t length) {
		return ~CRC32Implementation(data, length, ~0);
	}

	constexpr size_t strlen_c(const char* str) {
		return *str ? 1 + strlen_c(str + 1) : 0;
	}

	constexpr int WSID(const char* str) {
		return EncodeCRC32((uint8_t*)str, strlen_c(str));
	}

}

#define MAKE_HASHABLE(type, ...) \
namespace std {\
    template<> struct hash<type> {\
        std::size_t operator()(const type &t) const {\
            std::size_t ret = 0;\
            EmptySource::HashCombine(ret, __VA_ARGS__);\
            return ret;\
        }\
    };\
}