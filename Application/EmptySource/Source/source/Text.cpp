#include "../include/Text.h"

#ifdef WIN32
#include <Windows.h>
#endif

Char * WCharToChar(const WChar * From) {
    if (From == NULL) return NULL;
    
#ifdef WIN32
    int SizeNeeded = WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)std::wcslen(From), NULL, 0, NULL, NULL);
    Char* To = new Char[SizeNeeded];
    WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)std::wcslen(From), &To[0], SizeNeeded, NULL, NULL);
#else
    std::mbstate_t State = std::mbstate_t();
    size_t SizeNeeded = 1 + std::wcsrtombs(NULL, &From, 0, &State);
    Char* To = new Char[SizeNeeded];
    std::wcsrtombs(&To[0], &From, SizeNeeded, &State);
#endif
    
    return To;
}

WChar * CharToWChar(const Char * From) {
    if (From == NULL) return NULL;
    
#ifdef WIN32
    int SizeNeeded = MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)std::strlen(From), NULL, 0);
    WChar* To = (wchar_t*)LocalAlloc(LMEM_ZEROINIT, sizeof(wchar_t) * SizeNeeded + 1);
    MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)std::strlen(From) + 1, &To[0], SizeNeeded);
#else
    std::mbstate_t State = std::mbstate_t();
    size_t SizeNeeded = 1 + std::mbsrtowcs(NULL, &From, 0, &State);
    WChar* To = new WChar[SizeNeeded];
    std::mbsrtowcs(&To[0], &From, SizeNeeded, &State);
#endif
    
    return To;
}

String WStringToString(const WString & From) {
    if (From.empty()) return std::string();
    
#ifdef WIN32
    int SizeNeeded = WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)From.size(), NULL, 0, NULL, NULL);
    String To(SizeNeeded, 0);
    WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)From.size(), &To[0], SizeNeeded, NULL, NULL);
#else
    size_t SizeNeeded = 1 + std::wcstombs(NULL, &From[0], 0);
    String To = String(SizeNeeded, '\0');
    To.resize(std::wcstombs(&To[0], From.c_str(), SizeNeeded));
#endif
    
    return To;
}

WString StringToWString(const String & From) {
	if (From.empty()) return std::wstring();

#ifdef WIN32
	int SizeNeeded = MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)From.size(), NULL, 0);
	WString To(SizeNeeded, 0);
	MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)From.size(), &To[0], SizeNeeded);
#else
	size_t SizeNeeded = 1 + std::mbstowcs(NULL, &From[0], 0);
	WString To = WString(SizeNeeded, '\0');
	To.resize(std::mbstowcs(&To[0], From.c_str(), SizeNeeded));
#endif

	return To;
}