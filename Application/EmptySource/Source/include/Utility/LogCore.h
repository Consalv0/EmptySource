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

        WString LogText = L"";
        
#ifdef WIN32
        HANDLE hstdin = GetStdHandle(STD_INPUT_HANDLE); \
        HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE); \
        CONSOLE_SCREEN_BUFFER_INFO csbi; \
        GetConsoleScreenBufferInfo(hstdout, &csbi); \
        
        switch (Filter) {
            case LogNormal:   LogText += L"[LOG] "; break;
			case LogInfo:     LogText += L"[INFO] "; SetConsoleTextAttribute(hstdout, FOREGROUND_INTENSITY ); break;
            case LogWarning:  LogText += L"[WARNING] "; SetConsoleTextAttribute(hstdout, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY); break;
            case LogError:    LogText += L"[ERROR] "; SetConsoleTextAttribute(hstdout, FOREGROUND_RED | FOREGROUND_INTENSITY); break;
            case LogCritical: LogText += L"[CRITICAL] "; SetConsoleTextAttribute(hstdout, FOREGROUND_RED); break;
            case LogDebug:    LogText += L"[DEBUG] "; SetConsoleTextAttribute(hstdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY); break;
        }

		setlocale(LC_ALL, "en_US.UTF-8");
        std::wprintf((LogText + (Text + L"\n")).c_str(), Args ...);
        SetConsoleTextAttribute(hstdout, csbi.wAttributes);
#else
        
        switch (Filter) {
            case LogNormal:   LogText += L"\033[40m[LOG]\033[0m "; break;
			case LogInfo:     LogText += L"\033[90;40m[INFO]\033[90;49m "; break;
            case LogWarning:  LogText += L"\033[33;40m[WARNING]\033[33;49m "; break;
            case LogError:    LogText += L"\033[31;40m[ERROR]\033[31;49m "; break;
            case LogCritical: LogText += L"\033[31;40m[CRITICAL]\033[31;49m "; break;
            case LogDebug:    LogText += L"\033[32;40m[DEBUG]\033[32;49m "; break;
        }
        
        setlocale(LC_ALL, "en_US.UTF-8");
        std::wprintf((LogText + (Text + L"\033[0m\n")).c_str(), Args ...);
#endif
    }
}