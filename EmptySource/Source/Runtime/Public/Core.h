#pragma once

#if !defined(EMPTYSOURCE_CORE)
#define EMPTYSOURCE_CORE
#endif

#ifdef EMPTYSOURCE_CORE_LOG
#ifdef ES_ENABLE_ASSERTS
#define ES_ASSERT(X, ...) { if(!(X)) { LOG_CRITICAL("Assertion Failed: " __VA_ARGS__); __debugbreak(); } }
#define ES_CORE_ASSERT(X, ...) { if(!(X)) { LOG_CORE_CRITICAL("Assertion Failed: " __VA_ARGS__); __debugbreak(); } }
#else
#define ES_ASSERT(X, ...)
#define ES_CORE_ASSERT(X, ...)
#endif
#endif // EMPTYSOURCE_CORE_LOG

#include "Platform/Platform.h"
#include "CoreTypes.h"

