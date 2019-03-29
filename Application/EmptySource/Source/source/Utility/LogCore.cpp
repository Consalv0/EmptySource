
#include "../include/Utility/LogCore.h"

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
}
