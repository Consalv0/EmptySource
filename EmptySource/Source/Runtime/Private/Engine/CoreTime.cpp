
#include "Engine/Core.h"
#include "Rendering/Graphics.h"
#include "Engine/Application.h"
#include "Engine/CoreTime.h"
#ifndef __ANDROID__
#include <chrono>
#endif // !__ANDROID__
#include <ctime>

namespace EmptySource {

	unsigned long long Time::LastUpdateMicro = GetEpochTimeMicro();
	unsigned long long Time::LastDeltaMicro = 0;

	bool Time::bHasInitialized = false;

	unsigned int Time::TickCount = 0;
	unsigned long long Time::TickBuffer[MaxTickSamples];
	double Time::TickAverage = 30;

	void Time::Tick() {
		LastDeltaMicro = GetEpochTimeMicro() - LastUpdateMicro;
		LastUpdateMicro = GetEpochTimeMicro();

		TickBuffer[TickCount] = LastDeltaMicro;

		TickAverage = 0;
		for (int Count = 0; Count < MaxTickSamples; Count++) {
			TickAverage += TickBuffer[Count];
		}
		TickAverage = TickAverage / (double)MaxTickSamples;

		TickCount = (TickCount + 1) % MaxTickSamples;
	}

	float Time::GetDeltaTime() {
		return (float)LastDeltaMicro / 1000000.F;
	}

	double Time::GetDeltaTimeMilis() {
		return (double)LastDeltaMicro / 1000.0;
	}

	float Time::GetFrameRatePerSecond() {
		return float(1.0 / (TickAverage / 1000000.F));
	}

	unsigned long long Time::GetEpochTimeMicro() {
#ifndef __ANDROID__
		using namespace std::chrono;
		return time_point_cast<microseconds>(steady_clock::now()).time_since_epoch().count();
#else
		return 0;
#endif
	}

	float Time::GetEpochTimeSeconds() {
		return GetEpochTimeMicro() / 1000000.F;
	}

}