
#include "..\include\Core.h"
#include "..\include\Graphics.h"
#include "..\include\Application.h"
#include "..\include\Time.h"

#include <time.h>

unsigned long Time::LastUpdateTime = 0;
unsigned long Time::LastDeltaTime = 0;

unsigned Time::TickCount = 0;
unsigned long Time::TickBuffer[MaxTickSamples];
double Time::TickAverage = 60;

void Time::Tick() {
	LastDeltaTime = GetApplicationTime() - LastUpdateTime;
	LastUpdateTime = GetApplicationTime();
	
	TickCount = (TickCount + 1) % MaxTickSamples;
	TickBuffer[TickCount] = LastDeltaTime;

	TickAverage = 0;
	for (int Count = 0; Count < MaxTickSamples; Count++) {
		TickAverage += TickBuffer[Count] / 1000.0;
	}
	TickAverage = TickAverage / MaxTickSamples;
}

float Time::GetDeltaTime() {
	return LastDeltaTime / 1000.0F;
}

long Time::GetDeltaTimeMilis() {
	return LastDeltaTime;
}

float Time::GetFrameRate() {
	return float(1 / TickAverage);
}

unsigned long Time::GetApplicationTime() {
	SYSTEMTIME time;
	GetSystemTime(&time);
	return time.wHour * (long)3600000 +
		   time.wMinute * (long)60000 +
		   time.wSecond * (long)1000 +
		   time.wMilliseconds;
}

