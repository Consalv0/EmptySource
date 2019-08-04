
#include "Resources/ShaderStageResource.h"

namespace EmptySource {

	ShaderStageResource::ShaderStageResource(ShaderStageManager * Manager, const WString & Name, ShaderType Type, WString ShaderPath) :
		Supper((ResourceManager *)Manager, Name), Type(Type), ShaderPath(ShaderPath), ShaderCode("") {
	}

	ShaderStageResource::ShaderStageResource(ShaderStageManager * Manager, const WString & Name, ShaderType Type, const String & Code) :
		Supper((ResourceManager *)Manager, Name), Type(Type), ShaderPath(L""), ShaderCode(Code) {
	}

	void ShaderStageResource::Load() {
		if (Supper::LoadState == LS_Unloaded) {
			Supper::LoadState = LS_Loading;

			Resource = new ShaderStage(Type);
			if (FileManager::GetFile(ShaderPath) != NULL)
				Resource->CompileFromFile(ShaderPath);
			else
				Resource->CompileFromText(ShaderCode);

			if (Resource->IsValid()) {
				Supper::LoadState = LS_Loaded;
			}
			else {
				Supper::LoadState = LS_Unloaded;
			}
		}
	}

	void ShaderStageResource::Unload() {
		if (Supper::LoadState == LS_Loaded) {
			Supper::LoadState = LS_Unloading;
			Resource->Delete();
			delete Resource;
			Supper::LoadState = LS_Unloaded;
		}
	}

	void ShaderStageResource::Reload() {
		Unload();
		Load();
	}

}