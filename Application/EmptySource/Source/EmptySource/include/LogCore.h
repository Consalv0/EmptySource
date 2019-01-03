#pragma once

#include "..\include\Text.h"

constexpr unsigned char       NoLog = 0x00;
constexpr unsigned char         Log = 0x01;
constexpr unsigned char  LogWarning = 0x02;
constexpr unsigned char    LogError = 0x04;
constexpr unsigned char LogCritical = 0x08;
constexpr unsigned char    LogDebug = 0x16;

static unsigned char LogFilter = Log | LogWarning | LogError | LogCritical | LogDebug;

inline WString LogPrefix(unsigned char Filter, WString Text) {
	WString Prefix;

    setlocale(LC_ALL, "");
	
	switch (Filter) {
		case NoLog: Prefix = L""; break;
		case Log: Prefix = L"[LOG] "; break;
		case LogWarning: Prefix = L"[WARNING] "; break;
		case LogError: Prefix = L"[Error] "; break;
		case LogCritical: Prefix = L"[CRITICAL] "; break;
		case LogDebug: Prefix = L"[DEBUG] "; break;
		default: Prefix = L"[LOG] "; break;
	}

	return Prefix + (Text + L"\n");
}

#define _LOG(Filter, Text, ...) \
	if ((Filter & LogFilter) > 0 || Filter == NoLog) wprintf(LogPrefix(Filter, Text).c_str(), __VA_ARGS__)