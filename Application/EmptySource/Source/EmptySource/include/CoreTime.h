#pragma once

class Time {
public:
	friend class Application;
	
	// Don't use this unless you know what are you doing
	static void Tick();

	// Update the fixed update time
	// Don't use this unless you know what are you doing
	// static void FixedUpdate();

	// Time in seconds since the last frame;
	static float GetDeltaTime();

	// Time in milliseconds since the last frame;
	static double GetDeltaTimeMilis();

	// Get frame rate per second (FPS)
	static float GetFrameRatePerSecond();

	// Time since the app is running;
	static unsigned long long GetEpochTimeMili();

private:
	// Time since the last tick callback
	static unsigned long long LastUpdateMicro;
	static unsigned long long LastDeltaMicro;

	static bool bHasInitialized;

	static unsigned int TickCount;
	static double TickAverage;
	static const unsigned int MaxTickSamples = 25;
	static unsigned long long TickBuffer[MaxTickSamples];
};
