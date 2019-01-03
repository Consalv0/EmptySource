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
inline bool StringReplace(T& String, const T& From, const T& To) {
	size_t StartPos = String.find(From);

	if (StartPos == T::npos) {
		return false;
	}

	String.replace(StartPos, From.length(), To);
	return true;
}

// Create a string qith format
template<class T, typename ... Arguments>
inline std::basic_string<T> TextFormat(const std::basic_string<T>& Format, Arguments ... Args) {
	size_t Size = std::snprintf(nullptr, 0, Format.c_str(), Args ...) + 1; // Extra space for '\0'
	std::unique_ptr<T[]> Buffer(new T[Size]);
	std::snprintf(Buffer.get(), Size, Format.c_str(), Args ...);
	return std::basic_string<T>(Buffer.get(), Buffer.get() + Size - 1); // We don't want the '\0' inside
}