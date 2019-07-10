#pragma once

#include <string>

typedef std::string String;
typedef char Char;
typedef std::wstring WString;
typedef wchar_t WChar;

String WCharToString(const WChar* From);

WString CharToWString(const Char* From);

String WStringToString(const WString &From);

WString StringToWString(const String &From);

#include "../include/Utility/TextFormatting.h"