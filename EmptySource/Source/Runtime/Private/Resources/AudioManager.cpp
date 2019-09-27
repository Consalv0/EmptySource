
#include "CoreMinimal.h"
#include "Resources\AudioManager.h"

#include <yaml-cpp/yaml.h>
#include <SDL_audio.h>

namespace ESource {

	AudioSamplePtr AudioManager::GetAudioSample(const WString & Name) const {
		size_t UID = WStringToHash(Name);
		return GetAudioSample(UID);
	}

	AudioSamplePtr AudioManager::GetAudioSample(const size_t & UID) const {
		auto Resource = AudioSamplesList.find(UID);
		if (Resource != AudioSamplesList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	void AudioManager::FreeAudioSample(const WString & Name) {
		size_t UID = WStringToHash(Name);
		AudioSamplesList.erase(UID);
	}

	void AudioManager::AddAudioSample(const WString & Name, AudioSamplePtr Sample) {
		size_t UID = WStringToHash(Name);
		AudioSamplesList.insert({ UID, Sample });
	}

	void AudioManager::LoadAudioFromFile(const WString & Name, const WString & FilePath) {
		SDL_AudioSpec SampleSpecs;
		Uint32 BufferLength;
		Uint8 * SampleBuffer;

		FileStream * SampleFile = FileManager::GetFile(FilePath);

		if (SampleFile == NULL || SDL_LoadWAV(Text::WideToNarrow(SampleFile->GetPath()).c_str(), &SampleSpecs, &SampleBuffer, &BufferLength) == NULL) {
			LOG_ERROR("Couldn't not open the sound file or is invalid");
			return;
		}
		{
			static SDL_AudioCVT AudioConvert;
			if (SDL_BuildAudioCVT(&AudioConvert, SampleSpecs.format, SampleSpecs.channels, SampleSpecs.freq,
				AUDIO_F32LSB, 2, 48000))
			{
				AudioConvert.len = BufferLength;
				AudioConvert.buf = (Uint8 *)SDL_malloc(BufferLength * AudioConvert.len_mult);
				SDL_memcpy(AudioConvert.buf, SampleBuffer, BufferLength);
				SDL_ConvertAudio(&AudioConvert);
				SDL_FreeWAV(SampleBuffer);
				SampleSpecs.format = AUDIO_F32LSB;
				SampleBuffer = AudioConvert.buf;
				BufferLength = AudioConvert.len_cvt;
			}
		}

		AudioManager::AddAudioSample(Name, AudioSample::Create(SampleBuffer, SDL_AUDIO_BITSIZE(SampleSpecs.format) / 8, BufferLength, SampleSpecs.freq, SampleSpecs.channels));
	}

	void AudioManager::LoadResourcesFromFile(const WString & FilePath) {
	}

	AudioManager & AudioManager::GetInstance() {
		static AudioManager Manager;
		return Manager;
	}

}
