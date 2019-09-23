#pragma once

#if !defined(EMPTYSOURCE_CORE)
#define EMPTYSOURCE_CORE
#endif

#ifdef EMPTYSOURCE_CORE_LOG
#ifdef ES_ENABLE_ASSERTS
#define ES_ASSERT(x, ...) { if(!(x)) { LOG_CRITICAL("Assertion Failed: " __VA_ARGS__); __debugbreak(); } }
#define ES_CORE_ASSERT(x, ...) { if(!(x)) { LOG_CORE_CRITICAL("Assertion Failed: " __VA_ARGS__); __debugbreak(); } }
#else
#define ES_ASSERT(x, ...)
#define ES_CORE_ASSERT(x, ...)
#endif
#endif // EMPTYSOURCE_CORE_LOG

#include "Platform/Platform.h"
#include "CoreTypes.h"

