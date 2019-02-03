#include "..\Source\EmptySource\include\Utility\Timer.h"

Debug::Timer::Timer() {
	long StartTime = 0;
	long EndTime = 0;
}

void Debug::Timer::Start() {
	StartTime = clock();
}

void Debug::Timer::Stop() {
	EndTime = clock();
}

long Debug::Timer::GetStart() const {
	return StartTime;
}

long Debug::Timer::GetEnd() const {
	return EndTime;
}

long Debug::Timer::GetEnlapsed() const {
	return EndTime - StartTime;
}

float Debug::Timer::GetEnlapsedSeconds() const {
	return float(GetEnlapsed()) / CLOCKS_PER_SEC;
}
