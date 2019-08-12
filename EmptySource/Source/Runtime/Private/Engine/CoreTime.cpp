
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Engine/Application.h"
#include "Engine/CoreTime.h"
#include <chrono>
#include <ctime>

namespace EmptySource {

	unsigned long long Time::LastUpdateMicro = GetEpochTimeMicro();
	unsigned long long Time::LastDeltaMicro = 0;

	bool Time::bHasInitialized = false;

	unsigned int Time::TickCount = 0;
	unsigned long long Time::TickBuffer[MaxTickSamples];
	unsigned long long Time::TickAverage = 30;

	void Time::Tick() {
		LastDeltaMicro = GetEpochTimeMicro() - LastUpdateMicro;
		LastUpdateMicro = GetEpochTimeMicro();

		TickBuffer[TickCount] = LastDeltaMicro;

		TickAverage = 0;
		for (int Count = 0; Count < MaxTickSamples; Count++) {
			TickAverage += TickBuffer[Count];
		}
		TickAverage = TickAverage / MaxTickSamples;

		TickCount = (TickCount + 1) % MaxTickSamples;
	}

	Timestamp Time::GetTimeStamp() {
		return Timestamp(LastUpdateMicro, Time::GetEpochTimeMicro());
	}

	unsigned long long Time::GetEpochTimeMicro() {
		using namespace std::chrono;
		return time_point_cast<microseconds>(steady_clock::now()).time_since_epoch().count();
		return 0;
	}

	void Timestamp::Begin() {
		LastEpochTime = Time::GetEpochTime<Time::Micro>();
	}

	void Timestamp::Stop() {
		NowEpochTime = Time::GetEpochTime<Time::Micro>();
	}

}