#pragma once

#include "../include/Text.h"

#ifdef WIN32
#include <Windows.h>
#endif

namespace Debug {
    constexpr unsigned char       NoLog = 0;
    constexpr unsigned char   LogNormal = 1 << 0;
    constexpr unsigned char  LogWarning = 1 << 1;
    constexpr unsigned char    LogError = 1 << 2;
    constexpr unsigned char LogCritical = 1 << 3;
	constexpr unsigned char     LogInfo = 1 << 4;
	constexpr unsigned char    LogDebug = 1 << 5;
    
	struct LogFilter {
		static unsigned char Value;

		static void AddFilter(const unsigned char& Filter);
		static void RemoveFilter(const unsigned char& Filter);
		static void SetFilter(const unsigned char& Filter);
	};

	void LogClearLine(unsigned char Filter);

	template<typename ... Arguments>
	void LogUnadorned(unsigned char Filter, WString Text, Arguments ... Args) {
		if ((Filter & LogFilter::Value) == NoLog) {
			return;
		}

		setlocale(LC_ALL, "en_US.UTF-8");
		std::wprintf(Text.c_str(), Args ...);
	}

    template<typename ... Arguments>
    void Log(unsigned char Filter, WString Text, Arguments ... Args) {
		if ((Filter & LogFilter::Value) == NoLog) {
			return;
		}

        WString LogText;
        
#ifdef WIN32
        HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(hstdout, &csbi);
        
        if ((Filter & LogNormal & LogFilter::Value)   != NoLog) { LogText = L"[LOG] "; }
		if ((Filter & LogInfo & LogFilter::Value)     != NoLog) { LogText = L"[INFO] ";     SetConsoleTextAttribute(hstdout, FOREGROUND_INTENSITY); }
        if ((Filter & LogDebug & LogFilter::Value)    != NoLog) { LogText = L"[DEBUG] ";    SetConsoleTextAttribute(hstdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY); }
		if ((Filter & LogWarning & LogFilter::Value)  != NoLog) { LogText = L"[WARNING] ";  SetConsoleTextAttribute(hstdout, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY); }
		if ((Filter & LogError & LogFilter::Value)    != NoLog) { LogText = L"[ERROR] ";    SetConsoleTextAttribute(hstdout, FOREGROUND_RED | FOREGROUND_INTENSITY); }
		if ((Filter & LogCritical & LogFilter::Value) != NoLog) { LogText = L"[CRITICAL] "; SetConsoleTextAttribute(hstdout, FOREGROUND_RED); }

		setlocale(LC_ALL, "en_US.UTF-8");
        std::wprintf((LogText + (Text + L"\n")).c_str(), Args ...);
        SetConsoleTextAttribute(hstdout, csbi.wAttributes);
#else
        
        if ((Filter & LogNormal & LogFilter::Value)   != NoLog) LogText = L"\033[40m[LOG]\033[0m ";
		if ((Filter & LogInfo & LogFilter::Value)     != NoLog) LogText = L"\033[90;40m[INFO]\033[90;49m ";
        if ((Filter & LogDebug & LogFilter::Value)    != NoLog) LogText = L"\033[32;40m[DEBUG]\033[32;49m ";
        if ((Filter & LogWarning & LogFilter::Value)  != NoLog) LogText = L"\033[31;40m[ERROR]\033[31;49m ";
        if ((Filter & LogError & LogFilter::Value)    != NoLog) LogText = L"\033[31;40m[CRITICAL]\033[31;49m ";
        if ((Filter & LogCritical & LogFilter::Value) != NoLog) LogText = L"\033[33;40m[WARNING]\033[33;49m ";

        setlocale(LC_ALL, "en_US.UTF-8");
        std::wprintf((LogText + (Text + L"\033[0m\n")).c_str(), Args ...);
#endif
    }
}
