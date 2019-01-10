#pragma once

#include <string>
#include <locale> 
#include <codecvt>
#include <iostream>
#include <cstdio>
#include <memory>

typedef std::string String;
typedef char Char;
typedef std::wstring WString;
typedef wchar_t WChar;
// typedef std::wstring_convert<std::codecvt_utf8<WChar>> UTF8Convert;

// #define StringToWString(STRING) UTF8Convert().from_bytes(STRING)
// #define CharToWChar(STRING) UTF8Convert().from_bytes(STRING).c_str()
// #define WStringToString(STRING) UTF8Convert().to_bytes(STRING)
// #define WCharToChar(STRING) UTF8Convert().to_bytes(STRING).c_str()

inline Char* WCharToChar(const WChar* From) {
	if (From == NULL) return NULL;
	int SizeNeeded = WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)std::wcslen(From), NULL, 0, NULL, NULL);
	Char* To = new Char[SizeNeeded];
	WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)std::wcslen(From), &To[0], SizeNeeded, NULL, NULL);
	return To;
}

inline WChar* CharToWChar(const Char* From) {
	if (From == NULL) return NULL;
	int SizeNeeded = MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)std::strlen(From), NULL, 0);
	WChar* To = new WChar[SizeNeeded];
	MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)std::strlen(From), &To[0], SizeNeeded);
	return To;
}

inline String WStringToString(const WString &From) {
	if (From.empty()) return std::string();
	int SizeNeeded = WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)From.size(), NULL, 0, NULL, NULL);
	std::string To(SizeNeeded, 0);
	WideCharToMultiByte(CP_UTF8, 0, &From[0], (int)From.size(), &To[0], SizeNeeded, NULL, NULL);
	return To;
}

inline WString StringToWString(const String &From) {
	if (From.empty()) return std::wstring();
	int SizeNeeded = MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)From.size(), NULL, 0);
	std::wstring To(SizeNeeded, 0);
	MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)From.size(), &To[0], SizeNeeded);
	return To;
}


// Replace part of string with another string
template<class T>
inline bool TextReplace(T& String, const T& From, const T& To) {
	size_t StartPos = String.find(From);

	if (StartPos == T::npos) {
		return false;
	}

	String.replace(StartPos, From.length(), To);
	return true;
}

template<typename ... Arguments>
WString TextFormat(const WString& Format, Arguments ... Args) {
	const WChar* FormatBuffer = Format.c_str();
	int Size = (int)sizeof(WChar) * (int)Format.size();
	std::unique_ptr<WChar[]> Buffer;

	while (true) {
		Buffer = std::make_unique<WChar[]>(Size);
		int OldSize = Size;
		Size = std::swprintf(Buffer.get(), Size, FormatBuffer, Args ...);

		if (Size < 0) {
			Size += OldSize + 10;
		} else {
			break;
		}
	}

	return WString(Buffer.get(), Buffer.get() + Size);
}

template<typename ... Arguments>
WString TextFormat(const WChar* Format, Arguments ... Args) {
	int Size = (int)std::wcslen(Format);
	std::unique_ptr<WChar[]> Buffer;

	while (true) {
		Buffer = std::make_unique<WChar[]>(Size);
		int OldSize = Size;
		Size = std::swprintf(Buffer.get(), Size, Format, Args ...);

		if (Size < 0) {
			Size += OldSize + 25;
		} else {
			break;
		}
	}

	return WString(Buffer.get(), Buffer.get() + Size); // We don't want the '\0' inside
}