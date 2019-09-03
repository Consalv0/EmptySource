#pragma once

#include "CoreTypes.h"
#include "Audio/AudioSample.h"

namespace EmptySource {

	class AudioDevice {
	public:
		struct SamplePlayInfo {
			AudioSamplePtr Sample;
			bool bPause;
			bool bLoop;
			float Volume;
			unsigned int Pos;
			const size_t Identifier;
			
			SamplePlayInfo(AudioSamplePtr Sample, float Volume, bool Loop, bool Pause, const size_t & ID) :
				Sample(Sample), Volume(Volume), bLoop(Loop), bPause(Pause), Pos(0), Identifier(ID) {
			}
		};

		AudioDevice();

		~AudioDevice();

		size_t AddSample(AudioSamplePtr Sample, float Volume, bool Loop, bool PlayOnAdd);

		void RemoveSample(const size_t& Identifier);

		inline TDictionary<size_t, SamplePlayInfo *>::iterator begin() { return PlayInfoList.begin(); }
		inline TDictionary<size_t, SamplePlayInfo *>::iterator end() { return PlayInfoList.end(); }

	public:
		float Volume;

		unsigned char CurrentSample[32768];

	private:
		bool bInitialized;

		TDictionary<size_t, SamplePlayInfo *> PlayInfoList;
	};

}