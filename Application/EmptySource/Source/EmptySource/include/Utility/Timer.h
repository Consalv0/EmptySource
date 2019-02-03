#pragma once

#include <ctime>

namespace Debug {

	class Timer {
	private:
		long StartTime;
		long EndTime;

	public:
		Timer();

		void Start();
		void Stop();

		long GetStart() const;
		long GetEnd() const;

		long GetEnlapsed() const;
		float GetEnlapsedSeconds() const;
	};

}