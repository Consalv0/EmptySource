#pragma once

class Time {
public:
	friend class Application;
	// Don't use this unless you know what are you doing
	static void Tick();
	// Update the fixed update time
	// Don't use this unless you know what are you doing
	// static void FixedUpdate();

	// Time since the last frame;
	static float GetDeltaTime();

	// Time in milliseconds since the last frame;
	static long GetDeltaTimeMilis();

	// Get frame rate per second (FPS)
	static float GetFrameRate();

	// Time since the app is running;
	static unsigned long GetApplicationTime();

private:
	// Time since the last update callback
	static unsigned long LastUpdateTime;
	static unsigned long LastDeltaTime;

	static unsigned TickCount;
	static double TickAverage;
	static const unsigned MaxTickSamples = 25;
	static unsigned long TickBuffer[MaxTickSamples];
};