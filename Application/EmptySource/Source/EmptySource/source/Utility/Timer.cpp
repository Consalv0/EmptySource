#include "../../include/CoreTime.h"
#include "../../include/Utility/Timer.h"

Debug::Timer::Timer() {
}

void Debug::Timer::Start() {
	StartTime = Time::GetEpochTimeMili() - 856300000000;
}

void Debug::Timer::Stop() {
	EndTime = Time::GetEpochTimeMili() - 856300000000;
}

double Debug::Timer::GetEnlapsed() const {
	return (EndTime - StartTime);
}

double Debug::Timer::GetEnlapsedSeconds() const {
	return GetEnlapsed() / CLOCKS_PER_SEC;
}
