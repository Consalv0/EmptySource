
#include "Engine/Text.h"

#include <memory>
#include <locale>
#include <iostream>
#include <cstdio>
#include <stdio.h>

#ifdef WIN32
#include <Windows.h>
#endif

namespace EmptySource {

	String WCharToString(const WChar *From)
	{
		if (From == NULL)
			return NULL;

#ifdef WIN32
		int SizeNeeded = WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)std::wcslen(From), NULL, 0, NULL, NULL);
		String To = String(SizeNeeded, '\0');
		WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)std::wcslen(From), &To[0], SizeNeeded, NULL, NULL);
#else
		std::mbstate_t State = std::mbstate_t();
		size_t SizeNeeded = std::wcsrtombs(NULL, &From, 0, &State);
		String To = String(SizeNeeded, '\0');
		std::wcsrtombs(&To[0], &From, SizeNeeded, &State);
#endif

		return To;
	}

	WString CharToWString(const Char *From)
	{
		if (From == NULL)
			return NULL;

#ifdef WIN32
		int SizeNeeded = MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)std::strlen(From), NULL, 0);
		WString To = WString(SizeNeeded, L'\0');
		MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)std::strlen(From) + 1, &To[0], SizeNeeded);
#else
		std::mbstate_t State = std::mbstate_t();
		size_t SizeNeeded = std::mbsrtowcs(NULL, &From, 0, &State);
		WString To = WString(SizeNeeded, L'\0');
		std::mbsrtowcs(&To[0], &From, SizeNeeded, &State);
#endif

		return To;
	}

	String WStringToString(const WString &From)
	{
		if (From.empty())
			return std::string();

#ifdef WIN32
		int SizeNeeded = WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)From.size(), NULL, 0, NULL, NULL);
		String To = String(SizeNeeded, '\0');
		WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)From.size(), &To[0], SizeNeeded, NULL, NULL);
#else
		size_t SizeNeeded = std::wcstombs(NULL, &From[0], 0);
		String To = String(SizeNeeded, '\0');
		To.resize(std::wcstombs(&To[0], From.c_str(), SizeNeeded));
#endif

		return To;
	}

	WString StringToWString(const String &From)
	{
		if (From.empty())
			return std::wstring();

#ifdef WIN32
		int SizeNeeded = MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)From.size(), NULL, 0);
		WString To(SizeNeeded, 0);
		MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)From.size(), &To[0], SizeNeeded);
#else
		size_t SizeNeeded = std::mbstowcs(NULL, &From[0], 0);
		WString To = WString(SizeNeeded, '\0');
		To.resize(std::mbstowcs(&To[0], From.c_str(), SizeNeeded));
#endif

		return To;
	}

}