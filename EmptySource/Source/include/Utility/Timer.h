#pragma once

namespace Debug {

	// Util class to mesaure time between code
	class Timer {
	private:
		unsigned long long StartTime;
		unsigned long long EndTime;

	public:
		// Default Constructor
		Timer();

		// Save time stamp
		void Start();
		// Save end time stamp
		void Stop();

		// Get enlapsed time in miliseconds
		double GetEnlapsedMili() const;
		// Get enlapsed time in seconds
		double GetEnlapsedSeconds() const;
	};

}
