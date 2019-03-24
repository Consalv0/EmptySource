#include "../../include/CoreTime.h"
#include "../../include/Utility/Timer.h"

Debug::Timer::Timer() {
}

void Debug::Timer::Start() {
	StartTime = Time::GetEpochTimeMicro() - 856300000000;
}

void Debug::Timer::Stop() {
	EndTime = Time::GetEpochTimeMicro() - 856300000000;
}

double Debug::Timer::GetEnlapsedMili() const {
	return (EndTime - StartTime) / 1000.F;
}

double Debug::Timer::GetEnlapsedSeconds() const {
	return GetEnlapsedMili() / CLOCKS_PER_SEC;
}
