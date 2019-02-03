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
	WChar* To = (wchar_t*)LocalAlloc(LMEM_ZEROINIT, sizeof(wchar_t) * SizeNeeded + 1);
	MultiByteToWideChar(CP_UTF8, 0, &From[0], (int)std::strlen(From) + 1, &To[0], SizeNeeded);
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

namespace Text {
	// Replace part of string with another string
	template<class T>
	inline bool Replace(T& String, const T& From, const T& To) {
		size_t StartPos = String.find(From);

		if (StartPos == T::npos) {
			return false;
		}

		String.replace(StartPos, From.length(), To);
		return true;
	}

	template<typename ... Arguments>
	WString Formatted(const WString& Format, Arguments ... Args) {
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
	WString Formatted(const WChar* Format, Arguments ... Args) {
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

	template<class Num>
	inline WString FormattedUnit(const Num & Number, const int & Decimals) {
		double PrecisionNumber = (double)Number;
		WString Suffix = L"";
		if (PrecisionNumber > 1e3 && PrecisionNumber <= 1e6) {
			Suffix = L'k';
			PrecisionNumber /= 1e3;
		} else
		if (PrecisionNumber > 1e6 && PrecisionNumber <= 1e9) {
			Suffix = L'M';
			PrecisionNumber /= 1e6;
		} else
		if (PrecisionNumber > 1e9 && PrecisionNumber <= 1e12) {
			Suffix = L'G';
			PrecisionNumber /= 1e9;
		} else
		if (PrecisionNumber > 1e12) {
			Suffix = L'T';
			PrecisionNumber /= 1e12;
		}

		if (int(PrecisionNumber) == PrecisionNumber) {
			return Formatted(L"%d%s", (int)PrecisionNumber, Suffix.c_str());
		} else {
			return Formatted(L"%." + std::to_wstring(Decimals) + L"f%s", PrecisionNumber, Suffix.c_str());
		}
	}

	template<class Num>
	inline WString FormattedData(const Num & Number, const int & Decimals) {
		double PrecisionNumber = (double)Number;
		WString Suffix = L"b";
		if (PrecisionNumber > 1<<10 && PrecisionNumber <= 1<<20) {
			Suffix = L"kb";
			PrecisionNumber /= 1<<10;
		} else
		if (PrecisionNumber > 1<<20 && PrecisionNumber <= 1<<30) {
			Suffix = L"Mb";
			PrecisionNumber /= 1<<20;
		} else 
		if (PrecisionNumber > 1<<30 && PrecisionNumber <= (size_t)1<<40) {
			Suffix = L"Gb";
			PrecisionNumber /= 1<<30;
		} else
		if (PrecisionNumber > (size_t)1<<40) {
			Suffix = L"Tb";
			PrecisionNumber /= (size_t)1<<40;
		}

		if (int(PrecisionNumber) == PrecisionNumber) {
			return Formatted(L"%d%s", (int)PrecisionNumber, Suffix.c_str());
		} else {
			return Formatted(L"%." + std::to_wstring(Decimals) + L"f%s", PrecisionNumber, Suffix.c_str());
		}
	}
}