
#include "CoreMinimal.h"
#include "Audio/AudioSample.h"

namespace EmptySource {

	AudioSample::AudioSample(unsigned char * Buffer, unsigned int SampleSize, unsigned int BufferLength, unsigned int Frecuency, unsigned int ChannelCount) 
		: SampleSize(SampleSize), BufferLength(BufferLength), Frecuency(Frecuency), ChannelCount(ChannelCount), Format(EAudioFormat::Float32)
	{
		this->Buffer = new unsigned char[BufferLength];
		memcpy(this->Buffer, Buffer, BufferLength);
		Duration = (Time::Micro::ReturnType)(((BufferLength * 8u / (SampleSize * ChannelCount)) / (float)Frecuency) * Time::Second::GetSizeInMicro());
	}
	
	AudioSample::~AudioSample() {
		delete Buffer;
	}
	
	unsigned char * AudioSample::GetBufferAt(unsigned int Offset) {
		ES_CORE_ASSERT(BufferLength - Offset >= 0, "Trying to copy outside range of audio sample");
		return &Buffer[Offset];
	}

	unsigned char * AudioSample::GetBufferCopy(unsigned int Offset, unsigned int Length) const {
		ES_CORE_ASSERT(BufferLength - Offset > 0 || Length > BufferLength - Offset, "Trying to copy outside range of audio sample");
		unsigned char * ReturnVal = new unsigned char[Length];
		return (unsigned char *)memcpy(ReturnVal, &Buffer[Offset], Length);
	}

	bool AudioSample::SetData(unsigned char * InData, unsigned int Offset) {
		return false;
	}

	AudioSamplePtr AudioSample::Create(unsigned char * Buffer, unsigned int SampleSize, unsigned int BufferLength, unsigned int Frecuency, unsigned int ChannelCount) {
		return std::make_shared<AudioSample>(Buffer, SampleSize, BufferLength, Frecuency, ChannelCount);
	}

}

