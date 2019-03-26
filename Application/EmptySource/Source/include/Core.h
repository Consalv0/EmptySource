#pragma once

#define NOMINMAX

// Include windows headers
#ifdef WIN32
#include <windows.h>
#endif // _WIN32

// Include main standard headers
#include <stdexcept>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#include <io.h>
#endif
#include <fcntl.h>
#include <assert.h>

// Include standar headers
#include <vector>
#include <iomanip>
#ifdef WIN32
#include <conio.h>
#endif
#include <memory>

#include <string>
#include <locale> 
#include <codecvt>
#include <sstream>
#include <fstream>
#include <iostream>

#include "../include/CoreTypes.h"
#include "../include/Utility/LogCore.h"
