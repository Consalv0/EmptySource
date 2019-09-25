
#include "CoreMinimal.h"
#include "Utility/TextFormatting.h"

#include "Math/MathUtility.h"
#include "Math/Vector4.h"
#include "Math/Quaternion.h"
#include "Math/Matrix4x4.h"

#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Rendering/Rendering.h"

namespace EmptySource {

	Material::Material(const IName & Name) : Name(Name) {
		Shader = NULL;
		RenderPriority = 1000;
		bWriteDepth = true;
		DepthFunction = DF_LessEqual;
		FillMode = FM_Solid;
		CullMode = CM_CounterClockWise;
	}

	void Material::SetShaderProgram(const RShaderPtr & InShader) {
		ParameterLayout.Clear();
		if (InShader != NULL && InShader->IsValid()) {
			Shader = InShader;
			SetParameters(Shader->GetParameters());
		} else {
			LOG_CORE_ERROR(L"Trying to set shader '{}' in '{}' wich is not a valid program",
				InShader != NULL ? InShader->GetName().GetDisplayName().c_str() : L"NULL", Name.GetDisplayName().c_str());
			Shader.reset();
		}
	}

	RShaderPtr Material::GetShaderProgram() const {
		return Shader;
	}

	const IName & Material::GetName() const {
		return Name;
	}

	void Material::SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void * Data, const VertexBufferPtr & Buffer) const {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid())
			Program->GetProgram()->SetAttribMatrix4x4Array(AttributeName, Count, Data, Buffer);
	}

	void Material::SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count) const {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid())
			Program->GetProgram()->SetMatrix4x4Array(UniformName, Data, Count);
	}

	void Material::SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count) const {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid())
			Program->GetProgram()->SetFloat1Array(UniformName, Data, Count);
	}

	void Material::SetInt1Array(const NChar * UniformName, const int * Data, const int & Count) const {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid())
			Program->GetProgram()->SetInt1Array(UniformName, Data, Count);
	}

	void Material::SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count) const {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid())
			Program->GetProgram()->SetFloat2Array(UniformName, Data, Count);
	}

	void Material::SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count) const {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid())
			Program->GetProgram()->SetFloat3Array(UniformName, Data, Count);
	}

	void Material::SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count) const {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid())
			Program->GetProgram()->SetFloat4Array(UniformName, Data, Count);
	}

	void Material::SetTexture2D(const NChar * UniformName, RTexturePtr Text, const unsigned int & Position) const {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid() && Text->GetDimension() == ETextureDimension::Texture2D)
			Program->GetProgram()->SetTexture2D(UniformName, (Texture2D  *)Text->GetTexture(), Position);
	}

	void Material::SetTextureCubemap(const NChar * UniformName, RTexturePtr Text, const unsigned int & Position) const {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid() && Text->GetDimension() == ETextureDimension::Cubemap)
			Program->GetProgram()->SetTextureCubemap(UniformName, (Cubemap *)Text->GetTexture(), Position);
	}

	void Material::SetParameters(const TArray<ShaderParameters>& NewLayout) {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid())
			for (auto& Layout : NewLayout) {
				if (Program->GetProgram()->GetUniformLocation(Layout.Name.c_str()) != -1) {
					ParameterLayout.SetParameter(Layout);
				}
			}
	}

	void Material::AddParameters(const TArray<ShaderParameters>& NewLayout) {
		RShaderPtr Program = GetShaderProgram();
		if (Program != NULL && Program->IsValid())
			for (auto& Layout : NewLayout) {
				if (Program->GetProgram()->GetUniformLocation(Layout.Name.c_str()) != -1) {
					ParameterLayout.AddParameter(Layout);
				}
			}
	}

	void Material::Use() const {
		
		Rendering::SetActiveDepthTest(bWriteDepth);
		Rendering::SetDepthFunction(DepthFunction);
		Rendering::SetRasterizerFillMode(FillMode);
		Rendering::SetCullMode(CullMode);

		if (Shader && Shader->IsValid()) {
			Shader->GetProgram()->Bind();
			unsigned int i = 0;
			for (auto& Uniform : ParameterLayout) {
				switch (Uniform.Value.GetType()) {
				case EmptySource::EShaderUniformType::Matrix4x4Array:
					SetMatrix4x4Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.Matrix4x4Array.size()); break;
				case EmptySource::EShaderUniformType::Matrix4x4:
					SetMatrix4x4Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderUniformType::FloatArray:
					SetFloat1Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.FloatArray.size()); break;
				case EmptySource::EShaderUniformType::Float:
					SetFloat1Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderUniformType::Float2DArray:
					SetFloat2Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.Float2DArray.size()); break;
				case EmptySource::EShaderUniformType::Float2D:
					SetFloat2Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderUniformType::Float3DArray:
					SetFloat3Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.Float3DArray.size()); break;
				case EmptySource::EShaderUniformType::Float3D:
					SetFloat3Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderUniformType::Float4DArray:
					SetFloat3Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.Float3DArray.size()); break;
				case EmptySource::EShaderUniformType::Float4D:
					SetFloat4Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderUniformType::IntArray:
					SetInt1Array(Uniform.Name.c_str(), (int *)Uniform.Value.PointerToValue(), (int)Uniform.Value.IntArray.size()); break;
				case EmptySource::EShaderUniformType::Int:
					SetInt1Array(Uniform.Name.c_str(), (int *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderUniformType::Texture2D:
					SetTexture2D(Uniform.Name.c_str(), Uniform.Value.Texture, i); i++; break;
				case EmptySource::EShaderUniformType::Cubemap:
					SetTextureCubemap(Uniform.Name.c_str(), Uniform.Value.Texture, i); i++; break;
				case EmptySource::EShaderUniformType::None:
				default:
					break;
				}
			}
		}
	}

	bool Material::operator>(const Material & Other) const {
		return RenderPriority > Other.RenderPriority;
	}

	void MaterialLayout::SetParameter(const ShaderParameters & Variable) {
		auto Iterator = Find(Variable.Name);
		if (Iterator == end())
			MaterialVariables.push_back(Variable);
		else
			(*Iterator) = Variable.Value;
	}

	void MaterialLayout::AddParameter(const ShaderParameters & Property) {
		auto Iterator = Find(Property.Name);
		if (Iterator == end())
			MaterialVariables.push_back(Property);
	}

	TArray<ShaderParameters>::const_iterator MaterialLayout::Find(const NString & Name) const {
		return std::find_if(begin(), end(), [&Name](const ShaderParameters& Var) { return Var.Name == Name; });
	}

	TArray<ShaderParameters>::iterator MaterialLayout::Find(const NString & Name) {
		return std::find_if(begin(), end(), [&Name](const ShaderParameters& Var) { return Var.Name == Name; });
	}

}