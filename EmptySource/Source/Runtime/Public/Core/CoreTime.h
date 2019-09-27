#pragma once

namespace EmptySource {

	class Time;

	struct Timestamp {
	public:
		Timestamp() {
		}

		Timestamp(unsigned long long Last, unsigned long long Now)
			: LastEpochTime(Last), NowEpochTime(Now){
		};

	public:
		template<typename T>
		inline typename T::ReturnType GetDeltaTime() const { 
			return (NowEpochTime - LastEpochTime) / (typename T::ReturnType)T::GetSizeInMicro();
		};

		void Begin();

		void Stop();

		unsigned long long GetLastEpoch() const { return LastEpochTime; };
		unsigned long long GetNowEpoch() const { return NowEpochTime; };

		Timestamp operator+(const Timestamp& Other);

	private:
		unsigned long long LastEpochTime;
		unsigned long long NowEpochTime;
	};

	class Time {
	public:
		template<unsigned long long Size, typename Type>
		struct Duration { static constexpr unsigned long long GetSizeInMicro() { return Size; }; using ReturnType = Type; };

		using Micro  = Duration<1, unsigned long long>;
		using Mili   = Duration<1000, double>;
		using Second = Duration<1000000, float>;
		using Minute = Duration<166666667, float>;

		static unsigned long long MaxDeltaMicro;

	private:
		friend class Application;

		// Don't use this unless you know what are you doing
		static void Tick();

		// Update the fixed update time
		// Don't use this unless you know what are you doing
		// static void FixedTick();

		// Time since the last tick callback
		static unsigned long long LastUpdateMicro;
		static unsigned long long LastDeltaMicro;

		static bool bHasInitialized;

		static unsigned int TickCount;
		static unsigned long long TickAverage;
		static const unsigned int MaxTickSamples = 25;
		static unsigned long long TickBuffer[MaxTickSamples];

		static unsigned long long Time::GetEpochTimeMicro();

	public:

		static Timestamp GetTimeStamp();

		// Time in seconds since the last frame;
		template<typename T>
		static inline typename T::ReturnType GetDeltaTime() {
			return (LastDeltaMicro) / (typename T::ReturnType)T::GetSizeInMicro();
		}

		// Get the application tick average
		template<typename T>
		static inline typename T::ReturnType GetAverageDelta() {
			return (typename T::ReturnType)TickAverage / (typename T::ReturnType)T::GetSizeInMicro();
		}

		// Get the application frame rate
		template<typename T>
		static inline typename T::ReturnType GetFrameRate() {
			return (typename T::ReturnType)(1) / ((typename T::ReturnType)TickAverage / (typename T::ReturnType)T::GetSizeInMicro());
		}

		// Machine Time
		template<typename T>
		static inline typename T::ReturnType GetEpochTime() {
			return GetEpochTimeMicro() / (typename T::ReturnType)T::GetSizeInMicro();
		}
	};

}