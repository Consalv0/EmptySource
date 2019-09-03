#pragma once

#include "CoreTypes.h"

namespace EmptySource {

	enum class EAudioFormat {
		NotSupported, Float32
	};

	typedef std::shared_ptr<class AudioSample> AudioSamplePtr;

	class AudioSample {
	public:
		AudioSample(unsigned char * Buffer, unsigned int SampleSize, unsigned int BufferLength, unsigned int Frecuency, unsigned int ChannelCount);

		~AudioSample();

		template<typename T>
		inline typename T::ReturnType GetDuration() const { return (typename T::ReturnType)Duration / (typename T::ReturnType)T::GetSizeInMicro(); }

		template<typename T>
		inline typename T::ReturnType GetDurationAt(unsigned int Pos) const { 
			return (typename T::ReturnType)(BufferLength - Pos) / BufferLength * Duration / (typename T::ReturnType)T::GetSizeInMicro();
		}

		//* The number of channels in the audio.
		inline unsigned int GetChannelCount() const { return ChannelCount; }
		
		//* The sample frequency of the audio in Hertz
		inline unsigned int GetFrecuency() const { return Frecuency; }

		//* The length of one audio sample in bytes.
		inline unsigned int GetSampleSize() const { return SampleSize; }

		//* The length of the audio in bytes.
		inline unsigned int GetBufferLength() const { return BufferLength; }

		//* The length of the audio in samples.
		inline unsigned int GetSampleLength() const { return BufferLength / SampleSize; }

		inline EAudioFormat GetAudioFormat() const { return EAudioFormat::Float32; }

		unsigned char * GetBufferAt(unsigned int Offset);

		unsigned char * GetBufferCopy(unsigned int Offset, unsigned int Length) const;

		bool SetData(unsigned char * InData, unsigned int Offset);

		static AudioSamplePtr Create(unsigned char * Buffer, unsigned int SampleSize, unsigned int BufferLength, unsigned int Frecuency, unsigned int ChannelCount);

	private:
		unsigned long long Duration;
		unsigned int ChannelCount;
		unsigned int Frecuency;
		unsigned int SampleSize;
		unsigned int BufferLength;
		EAudioFormat Format;

		unsigned char * Buffer;
	};

}