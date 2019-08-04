#pragma once

namespace EmptySource {

	class AudioSample {
	private:
		float Duration;
		unsigned int ChannelCount;
		unsigned int Frecuency;
		unsigned int SampleSize;
		unsigned int SampleLength;

		unsigned char * Buffer;

	public:

		float GetDuration() const { return Duration; }

		unsigned int GetChannelCount() const { return ChannelCount; }

		unsigned int GetFrecuency() const { return Frecuency; }

		unsigned int GetSampleSize() const { return SampleSize; }

		unsigned int GetSampleLength() const { return SampleLength; }

		bool GetData(unsigned char * OutData, unsigned int Offset);

		bool SetData(unsigned char * InData, unsigned int Offset);
	};

}