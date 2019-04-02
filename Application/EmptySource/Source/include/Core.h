#pragma once

#define NOMINMAX

// Include windows headers
#if defined(_WIN32) && defined(_MSC_VER)
#include <windows.h>
#include <io.h>
#include <conio.h>
#endif // _WIN32

// Include main standard headers
#include <stdexcept>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <assert.h>

// Include standar headers
#include <vector>
#include <iomanip>
#include <memory>

#include <string>
#include <locale> 
#include <codecvt>
#include <sstream>
#include <fstream>
#include <iostream>

#include "../include/CoreTypes.h"
#include "../include/Utility/LogCore.h"
