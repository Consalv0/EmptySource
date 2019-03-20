#pragma once

#include "..\include\Text.h"

#ifdef _WIN32
#include <Windows.h>
#endif

#ifndef LOG_CORE
#define LOG_CORE

namespace Debug {
	constexpr unsigned char       NoLog = 0x00;
	constexpr unsigned char	  LogNormal = 0x01;
	constexpr unsigned char  LogWarning = 0x02;
	constexpr unsigned char    LogError = 0x04;
	constexpr unsigned char LogCritical = 0x08;
	constexpr unsigned char    LogDebug = 0x16;

	static unsigned char LogFilter = LogNormal | LogWarning | LogError | LogCritical | LogDebug;

	template<typename ... Arguments>
	inline void Log(unsigned char Filter, WString Text, Arguments ... Args) {
		if (Filter == NoLog) {
			std::wprintf(Text.c_str(), Args ...);
			return;
		}

		WString LogText = L"";

#ifdef WIN32
		HANDLE hstdin = GetStdHandle(STD_INPUT_HANDLE); \
		HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE); \
		CONSOLE_SCREEN_BUFFER_INFO csbi; \
		GetConsoleScreenBufferInfo(hstdout, &csbi); \

		switch (Filter) {
			case LogNormal: LogText += L"[LOG] "; break;
			case LogWarning: LogText += L"[WARNING] "; SetConsoleTextAttribute(hstdout, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY); break;
			case LogError: LogText += L"[Error] "; SetConsoleTextAttribute(hstdout, FOREGROUND_RED | FOREGROUND_INTENSITY); break;
			case LogCritical: LogText += L"[CRITICAL] "; SetConsoleTextAttribute(hstdout, FOREGROUND_RED); break;
			case LogDebug: LogText += L"[DEBUG] "; SetConsoleTextAttribute(hstdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY); break;
		}

		setlocale(LC_ALL, "");

		std::wprintf((LogText + (Text + L"\n")).c_str(), Args ...);
		SetConsoleTextAttribute(hstdout, csbi.wAttributes);
#endif
	}
}

#endif // !LOG_CORE
