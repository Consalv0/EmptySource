#pragma once

#include "../include/Text.h"

#ifdef WIN32
#include <Windows.h>
#endif

#ifndef LOG_CORE
#define LOG_CORE

namespace Debug {
    constexpr unsigned char       NoLog = 0x00;
    constexpr unsigned char    LogNormal = 0x01;
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
#else
        
        switch (Filter) {
            case LogNormal: LogText += L"\033[40m[LOG]\033[0m "; break;
            case LogWarning: LogText += L"\033[33;40m[WARNING]\033[33;49m "; break;
            case LogError: LogText += L"\033[31;40m[Error]\033[31;49m "; break;
            case LogCritical: LogText += L"\033[31;40m[CRITICAL]\033[31;49m "; break;
            case LogDebug: LogText += L"\033[32;40m[DEBUG]\033[32;49m "; break;
        }
        
        setlocale(LC_ALL, "en_US.UTF-8");
        std::wprintf((LogText + (Text + L"\033[0m\n")).c_str(), Args ...);
#endif
    }
}

#endif // !LOG_CORE
