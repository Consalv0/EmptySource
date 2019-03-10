
#include "..\include\Core.h"
#include "..\include\Graphics.h"
#include "..\include\Application.h"
#include "..\include\Time.h"

#include <chrono>
#include <ctime>

unsigned long long Time::LastUpdateMicro = 0;
unsigned long long Time::LastDeltaMicro = 0;

unsigned int Time::TickCount = 0;
unsigned long long Time::TickBuffer[MaxTickSamples];
double Time::TickAverage = 60;

void Time::Tick() {
	LastDeltaMicro = GetEpochTimeMilli() - 856300000000 - LastUpdateMicro;
	LastUpdateMicro = GetEpochTimeMilli() - 856300000000;
	
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

unsigned long long Time::GetEpochTimeMilli() {
	using namespace std::chrono;
	steady_clock::time_point Now = high_resolution_clock::now();
	return duration_cast<std::chrono::microseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

