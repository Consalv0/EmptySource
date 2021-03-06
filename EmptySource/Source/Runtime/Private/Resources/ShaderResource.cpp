
#include "CoreMinimal.h"
#include "Resources/ShaderManager.h"
#include "Resources/ShaderResource.h"
#include "Resources/TextureManager.h"

#include <yaml-cpp/yaml.h>

namespace ESource {

	bool RShader::IsValid() const {
		return LoadState == LS_Loaded && ShaderPointer->IsValid();
	}

	void RShader::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			LOG_CORE_DEBUG(L"Loading Shader {}...", Name.GetDisplayName().c_str());
			if (!Origin.empty()) {
				FileStream * ShaderFile = FileManager::GetFile(Origin);
				if (ShaderFile == NULL) {
					LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}
				SourceCode.clear();
				if (!ShaderFile->ReadNarrowStream(&SourceCode)) {
					LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}
				ShaderFile->Close();
			}
		}
		LoadState = LoadFromShaderSource(SourceCode) ? LS_Loaded : LS_Unloaded;
	}

	void RShader::LoadAsync() {
		ES_CORE_ASSERT(true, "Not implemented")
	}

	void RShader::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		for (auto& Stage : Stages)
			delete Stage;
		delete ShaderPointer;
		ShaderPointer = NULL;
		Stages.clear();
		LoadState = LS_Unloaded;
	}

	void RShader::Reload() {
		Unload();
		Load();
	}

	void RShader::SetParameters(const TArrayInitializer<ShaderParameter> & InProperties) {
		Parameters = InProperties;
	}

	void RShader::SetParameters(const TArray<ShaderParameter>& InProperties) {
		Parameters = InProperties;
	}

	RShader::RShader(const IName & Name, const WString & Origin, const NString& Source, int CompileFlags)
		: ResourceHolder(Name, Origin), Parameters(), SourceCode(Source), Stages(), CompileFlags(CompileFlags) {
	}

	bool RShader::LoadFromShaderSource(const NString & FileInfo) {
		YAML::Node BaseNode; {
			try {
				BaseNode = YAML::Load(FileInfo.c_str());
			} catch (...) {
				return false;
			}
		}

		///// TODO: This may be retrieved from the renderer in the future
		YAML::Node CodeNode = BaseNode["GLSL"];

		YAML::Node NameNode = BaseNode["Name"];
		YAML::Node ParametersNode = BaseNode["Parameters"];
		YAML::Node StagesNode = CodeNode["Stages"];

		WString ShaderName = L"ShaderProgram";
		if (NameNode.IsDefined()) ShaderName = Text::NarrowToWide(NameNode.as<NString>());

		if (CodeNode.IsDefined()) {
			if (StagesNode.IsDefined()) {
				for (YAML::iterator Iterator = StagesNode.begin(); Iterator != StagesNode.end(); ++Iterator) {
					const YAML::Node& Stage = *Iterator;
					NString Type = Stage["StageType"].as<NString>();

					if (Stage["Code"].IsDefined()) {
						Stages.push_back( ShaderStage::CreateFromText(
							Stage["Code"].as<NString>(),
							ShaderManager::StringToStageType(Type),
							Stage["Code"].Mark().line,
							CompileFlags)
						);
					}
				}
			}
		}
		ShaderPointer = ShaderProgram::Create(Stages);

		if (ParametersNode.IsDefined()) {
			TArray<ShaderParameter> Parameters;
			for (auto & Uniform : ParametersNode) {
				NString UniformName = Uniform["Uniform"].as<NString>();
				NString TypeName = Uniform["Type"].as<NString>();
				int Flags = SPFlags_None;
				if (Uniform["IsColor"]) {
					Flags |= Uniform["IsColor"].as<bool>() ? SPFlags_IsColor : SPFlags_None;
				}
				if (TypeName == "None")           Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(), Flags);
				else if (TypeName == "Matrix4x4Array") Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(TArray<Matrix4x4>()), Flags);
				else if (TypeName == "Matrix4x4")      Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(Matrix4x4()), Flags);
				else if (TypeName == "FloatArray")     Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(TArray<float>()), Flags);
				else if (TypeName == "Float")          Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(Uniform["DefaultValue"].as<float>()), Flags);
				else if (TypeName == "Float2DArray")   Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(TArray<Vector2>()), Flags);
				else if (TypeName == "Float2D")        Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(Vector2(
					Uniform["DefaultValue"][0].as<float>(), Uniform["DefaultValue"][1].as<float>())
				), Flags);
				else if (TypeName == "Float3DArray")   Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(TArray<Vector3>()), Flags);
				else if (TypeName == "Float3D")        Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(Vector3(
					Uniform["DefaultValue"][0].as<float>(), Uniform["DefaultValue"][1].as<float>(), Uniform["DefaultValue"][2].as<float>())
				), Flags);
				else if (TypeName == "Float4DArray")   Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(TArray<Vector4>()), Flags);
				else if (TypeName == "Float4D")        Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(Vector4(
					Uniform["DefaultValue"][0].as<float>(), Uniform["DefaultValue"][1].as<float>(), Uniform["DefaultValue"][2].as<float>(),
					Uniform["DefaultValue"][3].as<float>())
				), Flags);
				else if (TypeName == "Texture2D")      Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(ETextureDimension::Texture2D,
					TextureManager::GetInstance().GetTexture(Text::NarrowToWide(Uniform["DefaultValue"].as<NString>()))
				), Flags);
				else if (TypeName == "Cubemap")        Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(ETextureDimension::Cubemap,
					NULL), Flags);
				else if (TypeName == "IntArray")       Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(TArray<int>()), Flags);
				else if (TypeName == "Int")            Parameters.emplace_back(UniformName, ShaderParameter::PropertyValue(0), Flags);
			}

			SetParameters(Parameters);
			return true;
		}

		return false;
	}

	RShader::~RShader() {
		Unload();
	}

}