
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Core/Application.h"
#include "Core/CoreTime.h"
#include <chrono>
#include <ctime>

namespace ESource {

	unsigned long long Time::LastUpdateMicro = GetEpochTimeMicro();
	unsigned long long Time::LastDeltaMicro = 0;

	bool Time::bHasInitialized = false;

	unsigned int Time::TickCount = 0;
	unsigned long long Time::TickBuffer[MaxTickSamples];
	unsigned long long Time::TickAverage = 30; 
	
	unsigned long long Time::MaxDeltaMicro = Time::Second::GetSizeInMicro() / 30;

	void Time::Tick() {
		unsigned long long TickTime = GetEpochTimeMicro();
		LastDeltaMicro = TickTime - LastUpdateMicro;

		if (LastDeltaMicro < MaxDeltaMicro) {
			unsigned long long Delta = MaxDeltaMicro - LastDeltaMicro;
			std::this_thread::sleep_for(std::chrono::microseconds(Delta - 1000));
		}

		LastUpdateMicro = GetEpochTimeMicro();
		LastDeltaMicro += LastUpdateMicro - TickTime;
		
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

	Timestamp Timestamp::operator+(const Timestamp & Other) {
		return Timestamp(Math::Max(LastEpochTime, Other.LastEpochTime), Math::Min(NowEpochTime, Other.NowEpochTime));
	}

}