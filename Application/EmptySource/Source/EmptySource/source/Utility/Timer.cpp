#include "../../include/Utility/Timer.h"

Debug::Timer::Timer() {
}

void Debug::Timer::Start() {
	StartTime = std::chrono::high_resolution_clock::now();
}

void Debug::Timer::Stop() {
	EndTime = std::chrono::high_resolution_clock::now();
}

double Debug::Timer::GetEnlapsed() const {
	std::chrono::duration<double, std::milli> Enlapsed = EndTime - StartTime;
	return Enlapsed.count();
}

double Debug::Timer::GetEnlapsedSeconds() const {
	return GetEnlapsed() / CLOCKS_PER_SEC;
}
