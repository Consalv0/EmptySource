#pragma once

#include <string>
#include <locale>
#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <memory>

typedef std::string String;
typedef char Char;
typedef std::wstring WString;
typedef wchar_t WChar;

Char* WCharToChar(const WChar* From);

WChar* CharToWChar(const Char* From);

String WStringToString(const WString &From);

WString StringToWString(const String &From);

#include "../include/Utility/TextFormatting.h"