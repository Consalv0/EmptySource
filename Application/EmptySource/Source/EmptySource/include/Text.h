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
typedef std::wstring_convert<std::codecvt_utf8<WChar>> UTF8Convert;

#define StringToWString(STRING) UTF8Convert().from_bytes(STRING)
#define CharToWChar(STRING) UTF8Convert().from_bytes(STRING).c_str()
#define WStringToString(STRING) UTF8Convert().to_bytes(STRING)
#define WCharToChar(STRING) UTF8Convert().to_bytes(STRING).c_str()

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