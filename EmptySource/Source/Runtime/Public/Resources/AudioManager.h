#pragma once

#include "Resources/ResourceManager.h"
#include "Audio/AudioSample.h"

namespace ESource {

	class AudioManager : public ResourceManager {
	public:
		AudioSamplePtr GetAudioSample(const WString& Name) const;

		AudioSamplePtr GetAudioSample(const size_t & UID) const;

		void FreeAudioSample(const WString& Name);

		void AddAudioSample(const WString& Name, AudioSamplePtr Sample);

		void LoadAudioFromFile(const WString& Name, const WString& FilePath);

		virtual inline EResourceType GetResourceType() const override { return RT_Audio; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		static AudioManager& GetInstance();

	private:
		TDictionary<size_t, AudioSamplePtr> AudioSamplesList;

	};

}