
#include "CoreMinimal.h"
#include "Audio/AudioSample.h"

namespace ESource {

	AudioSample::AudioSample(unsigned char * Buffer, uint32_t SampleSize, uint32_t BufferLength, uint32_t Frecuency, uint32_t ChannelCount) 
		: SampleSize(SampleSize), BufferLength(BufferLength), Frecuency(Frecuency), ChannelCount(ChannelCount), Format(EAudioFormat::Float32)
	{
		this->Buffer = new unsigned char[BufferLength];
		memcpy(this->Buffer, Buffer, BufferLength);
		Duration = (Time::Micro::ReturnType)(((BufferLength * 8u / (SampleSize * ChannelCount)) / (float)Frecuency) * Time::Second::GetSizeInMicro());
	}
	
	AudioSample::~AudioSample() {
		delete Buffer;
	}
	
	unsigned char * AudioSample::GetBufferAt(uint32_t Offset) {
		ES_CORE_ASSERT(BufferLength - Offset >= 0, "Trying to copy outside range of audio sample");
		return &Buffer[Offset];
	}

	unsigned char * AudioSample::GetBufferCopy(uint32_t Offset, uint32_t Length) const {
		ES_CORE_ASSERT(BufferLength - Offset > 0 || Length > BufferLength - Offset, "Trying to copy outside range of audio sample");
		unsigned char * ReturnVal = new unsigned char[Length];
		return (unsigned char *)memcpy(ReturnVal, &Buffer[Offset], Length);
	}

	bool AudioSample::SetData(unsigned char * InData, uint32_t Offset) {
		return false;
	}

	AudioSamplePtr AudioSample::Create(unsigned char * Buffer, uint32_t SampleSize, uint32_t BufferLength, uint32_t Frecuency, uint32_t ChannelCount) {
		return std::make_shared<AudioSample>(Buffer, SampleSize, BufferLength, Frecuency, ChannelCount);
	}

}

