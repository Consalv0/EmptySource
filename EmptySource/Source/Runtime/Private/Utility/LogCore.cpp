
#include "Utility/LogCore.h"

#ifdef ES_PLATFORM_WINDOWS
#include <Windows.h>
#endif

namespace EmptySource {

	namespace Debug {
		unsigned char LogFilter::Value = LogNormal | LogWarning | LogError | LogCritical | LogInfo | LogDebug;

		void LogFilter::AddFilter(const unsigned char& Filter) {
			Value |= Filter;
		}

		void LogFilter::RemoveFilter(const unsigned char& Filter) {
			Value ^= Filter;
		}

		void LogFilter::SetFilter(const unsigned char & Filter) {
			Value = Filter;
		}

		void LogClearLine(unsigned char Filter) {
			if ((Filter & LogFilter::Value) == NoLog) {
				return;
			}

			std::wprintf(L"\r");
		}

#ifdef ES_PLATFORM_WINDOWS
		unsigned short GetWin32TextConsoleColor() {
			static HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
			CONSOLE_SCREEN_BUFFER_INFO csbi;
			GetConsoleScreenBufferInfo(hstdout, &csbi);
			return csbi.wAttributes;
		}

		void SetWin32TextConsoleColor(const unsigned short & Att) {
			static HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(hstdout, Att);
		}
#endif
	}

}