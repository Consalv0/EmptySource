#pragma once

#include "CoreTypes.h"

namespace ESource {

	enum class EAudioFormat {
		NotSupported, Float32
	};

	typedef std::shared_ptr<class AudioSample> AudioSamplePtr;

	class AudioSample {
	public:
		AudioSample(unsigned char * Buffer, uint32_t SampleSize, uint32_t BufferLength, uint32_t Frecuency, uint32_t ChannelCount);

		~AudioSample();

		template<typename T>
		inline typename T::ReturnType GetDuration() const { return (typename T::ReturnType)Duration / (typename T::ReturnType)T::GetSizeInMicro(); }

		template<typename T>
		inline typename T::ReturnType GetDurationAt(uint32_t Pos) const { 
			return (typename T::ReturnType)(BufferLength - Pos) / BufferLength * Duration / (typename T::ReturnType)T::GetSizeInMicro();
		}

		//* The number of channels in the audio.
		inline uint32_t GetChannelCount() const { return ChannelCount; }
		
		//* The sample frequency of the audio in Hertz
		inline uint32_t GetFrecuency() const { return Frecuency; }

		//* The length of one audio sample in bytes.
		inline uint32_t GetSampleSize() const { return SampleSize; }

		//* The length of the audio in bytes.
		inline uint32_t GetBufferLength() const { return BufferLength; }

		//* The length of the audio in samples.
		inline uint32_t GetSampleLength() const { return BufferLength / SampleSize; }

		inline EAudioFormat GetAudioFormat() const { return EAudioFormat::Float32; }

		unsigned char * GetBufferAt(uint32_t Offset);

		unsigned char * GetBufferCopy(uint32_t Offset, uint32_t Length) const;

		bool SetData(unsigned char * InData, uint32_t Offset);

		static AudioSamplePtr Create(unsigned char * Buffer, uint32_t SampleSize, uint32_t BufferLength, uint32_t Frecuency, uint32_t ChannelCount);

	private:
		unsigned long long Duration;
		uint32_t ChannelCount;
		uint32_t Frecuency;
		uint32_t SampleSize;
		uint32_t BufferLength;
		EAudioFormat Format;

		unsigned char * Buffer;
	};

}