#pragma once

#include <ctime>

namespace Debug {

	// Util class to mesaure time between code
	class Timer {
	private:
		long StartTime;
		long EndTime;

	public:
		Timer();

		// --- Save time stamp
		void Start();
		// --- Save end time stamp
		void Stop();

		// --- Get start time stamp
		long GetStart() const;
		// --- Get stop time stamp
		long GetEnd() const;

		// --- Get enlapsed time in miliseconds
		long GetEnlapsed() const;
		// --- Get enlapsed time in seconds
		float GetEnlapsedSeconds() const;
	};

}