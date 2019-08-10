#pragma once

#include "Resources/ResourceHolder.h"
#include "Rendering/ShaderStage.h"

namespace EmptySource {

	struct ShaderStageResource : public ResourceHolder {
	protected:
		typedef ResourceHolder Supper;

		friend class ShaderStageManager;

		ShaderStage * Resource;

		EShaderType Type;

		WString ShaderPath;

		NString ShaderCode;

		ShaderStageResource(class ShaderStageManager * Manager, const WString & Name, EShaderType Type, WString ShaderPath);

		ShaderStageResource(class ShaderStageManager * Manager, const WString & Name, EShaderType Type, const NString & Code);

	public:

		void Load();

		void Unload();

		void Reload();

		const EShaderType & GetShaderType() const { return Type; };

		WString GetShaderPath() const { return ShaderPath; };
	};

}