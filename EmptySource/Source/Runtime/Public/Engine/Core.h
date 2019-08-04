#pragma once

#define NOMINMAX

// Include windows headers
#ifdef ES_PLATFORM_WINDOWS
#include <windows.h>
#include <io.h>
#include <conio.h>
#endif // ES_PLATFORM_WINDOWS

// Include main standard headers
#include <stdexcept>
#include <errno.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "Engine/CoreTypes.h"
#include "Utility/LogCore.h"
