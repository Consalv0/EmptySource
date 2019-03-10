#include "..\Source\EmptySource\include\Utility\Timer.h"

Debug::Timer::Timer() {
	std::chrono::steady_clock::time_point StartTime;
	std::chrono::steady_clock::time_point EndTime;
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
	std::chrono::duration<double, std::milli> Enlapsed = EndTime - StartTime;
	return GetEnlapsed() / CLOCKS_PER_SEC;
}
