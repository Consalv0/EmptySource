#pragma once

#include "Engine/CoreTypes.h"

namespace EmptySource {

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

#ifdef ES_PLATFORM_WINDOWS

		unsigned short GetWin32TextConsoleColor();

		void SetWin32TextConsoleColor(const unsigned short & Att);

#endif // ES_PLATFORM_WINDOWS

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

#ifdef ES_PLATFORM_WINDOWS
			unsigned short Att = GetWin32TextConsoleColor();

			if ((Filter & LogNormal & LogFilter::Value) != NoLog) { LogText = L"[LOG] "; }
			if ((Filter & LogInfo & LogFilter::Value) != NoLog) { LogText = L"[INFO] ";     SetWin32TextConsoleColor(0x0008); }
			if ((Filter & LogDebug & LogFilter::Value) != NoLog) { LogText = L"[DEBUG] ";    SetWin32TextConsoleColor(0x0002 | 0x0008); }
			if ((Filter & LogWarning & LogFilter::Value) != NoLog) { LogText = L"[WARNING] ";  SetWin32TextConsoleColor(0x0002 | 0x0004 | 0x0008); }
			if ((Filter & LogError & LogFilter::Value) != NoLog) { LogText = L"[ERROR] ";    SetWin32TextConsoleColor(0x0004 | 0x0008); }
			if ((Filter & LogCritical & LogFilter::Value) != NoLog) { LogText = L"[CRITICAL] "; SetWin32TextConsoleColor(0x0004); }

			setlocale(LC_ALL, "en_US.UTF-8");
			std::wprintf((LogText + (Text + L"\n")).c_str(), Args ...);
			SetWin32TextConsoleColor(Att);
#else

			if ((Filter & LogNormal & LogFilter::Value) != NoLog) LogText = L"\033[40m[LOG]\033[0m ";
			if ((Filter & LogInfo & LogFilter::Value) != NoLog) LogText = L"\033[90;40m[INFO]\033[90;49m ";
			if ((Filter & LogDebug & LogFilter::Value) != NoLog) LogText = L"\033[32;40m[DEBUG]\033[32;49m ";
			if ((Filter & LogWarning & LogFilter::Value) != NoLog) LogText = L"\033[33;40m[WARNING]\033[33;49m ";
			if ((Filter & LogError & LogFilter::Value) != NoLog) LogText = L"\033[31;40m[ERROR]\033[31;49m ";
			if ((Filter & LogCritical & LogFilter::Value) != NoLog) LogText = L"\033[33;40m[CRITICAL]\033[33;49m ";

			setlocale(LC_ALL, "en_US.UTF-8");
			std::wprintf((LogText + (Text + L"\033[0m\n")).c_str(), Args ...);
#endif
		}
	}

}