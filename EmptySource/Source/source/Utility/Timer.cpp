#include "../../include/CoreTime.h"
#include "../../include/Utility/Timer.h"

namespace EmptySource {

	Debug::Timer::Timer() {
	}

	void Debug::Timer::Start() {
		StartTime = Time::GetEpochTimeMicro();
	}

	void Debug::Timer::Stop() {
		EndTime = Time::GetEpochTimeMicro();
	}

	double Debug::Timer::GetEnlapsedMili() const {
		return double(EndTime - StartTime) / 1000.F;
	}

	double Debug::Timer::GetEnlapsedSeconds() const {
		return GetEnlapsedMili() / 1000.F;
	}

}