#pragma once

#include "Resources/ResourceHolder.h"
#include "Graphics/ShaderStage.h"

namespace EmptySource {

	struct ShaderStageResource : public ResourceHolder {
	protected:
		typedef ResourceHolder Supper;

		friend class ShaderStageManager;

		ShaderStage * Resource;

		ShaderType Type;

		WString ShaderPath;

		String ShaderCode;

		ShaderStageResource(class ShaderStageManager * Manager, const WString & Name, ShaderType Type, WString ShaderPath);

		ShaderStageResource(class ShaderStageManager * Manager, const WString & Name, ShaderType Type, const String & Code);

	public:

		void Load();

		void Unload();

		void Reload();

		const ShaderType & GetShaderType() const { return Type; };

		WString GetShaderPath() const { return ShaderPath; };
	};

}