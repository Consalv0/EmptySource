
#include "CoreMinimal.h"

#include "Math/MathUtility.h"
#include "Math/Vector4.h"
#include "Math/Quaternion.h"
#include "Math/Matrix4x4.h"

#include "Rendering/Texture.h"
#include "Rendering/Material.h"

#include "Rendering/GLFunctions.h"

namespace EmptySource {

	Material::Material(const WString & Name) : Name(Name) {
		MaterialShader = NULL;
		RenderPriority = 1000;
		bUseDepthTest = true;
		DepthFunction = DF_LessEqual;
		FillMode = FM_Solid;
		CullMode = CM_CounterClockWise;
	}

	void Material::SetShaderProgram(ShaderPtr Value) {
		if (Value != NULL && Value->IsValid()) {
			MaterialShader = Value;
		}
		else {
			LOG_CORE_ERROR(L"The Shader Program '{}' is not a valid program", Value != NULL ? Value->GetName() : L"NULL");
		}
	}

	ShaderPtr Material::GetShaderProgram() const {
		return MaterialShader;
	}

	WString Material::GetName() const {
		return Name;
	}

	void Material::SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void * Data, const VertexBufferPtr & Buffer) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetAttribMatrix4x4Array(AttributeName, Count, Data, Buffer);
	}

	void Material::SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetMatrix4x4Array(UniformName, Data, Count);
	}

	void Material::SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetFloat1Array(UniformName, Data, Count);
	}

	void Material::SetInt1Array(const NChar * UniformName, const int * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetInt1Array(UniformName, Data, Count);
	}

	void Material::SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetFloat2Array(UniformName, Data, Count);
	}

	void Material::SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetFloat3Array(UniformName, Data, Count);
	}

	void Material::SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetFloat4Array(UniformName, Data, Count);
	}

	void Material::SetTexture2D(const NChar * UniformName, TexturePtr Text, const unsigned int & Position) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetTexture2D(UniformName, Text, Position);
	}

	void Material::SetTextureCubemap(const NChar * UniformName, TexturePtr Text, const unsigned int & Position) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetTexture2D(UniformName, Text, Position);
	}

	void Material::Use() const {
		// --- Activate Z-buffer
		if (bUseDepthTest) {
			glEnable(GL_DEPTH_TEST);
		}
		else {
			glDisable(GL_DEPTH_TEST);
		}

		switch (DepthFunction) {
		case DF_Always:
			glDepthFunc(GL_ALWAYS); break;
		case DF_Equal:
			glDepthFunc(GL_EQUAL); break;
		case DF_Greater:
			glDepthFunc(GL_GREATER); break;
		case DF_GreaterEqual:
			glDepthFunc(GL_GEQUAL); break;
		case DF_Less:
			glDepthFunc(GL_LESS); break;
		case DF_LessEqual:
			glDepthFunc(GL_LEQUAL); break;
		case DF_Never:
			glDepthFunc(GL_NEVER); break;
		case DF_NotEqual:
			glDepthFunc(GL_NOTEQUAL); break;
		}

		if (CullMode == CM_None) {
			glDisable(GL_CULL_FACE);
		}
		else {
			glEnable(GL_CULL_FACE);
			switch (CullMode) {
			case CM_ClockWise:
				glCullFace(GL_FRONT); break;
			case CM_CounterClockWise:
				glCullFace(GL_BACK); break;
			case CM_None:
				break;
			}
		}

		switch (FillMode) {
		case FM_Point:
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); break;
		case FM_Wireframe:
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); break;
		case FM_Solid:
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); break;
		}

		if (MaterialShader && MaterialShader->IsValid()) {
			MaterialShader->Bind();
			unsigned int i = 0;
			for (auto& Uniform : VariableLayout) {
				switch (Uniform.VariableType) {
				case EmptySource::EMaterialDataType::Matrix4x4Array:
					SetMatrix4x4Array(Uniform.Name.c_str(), Uniform.Matrix4x4Array[0].PointerToValue(), (int)Uniform.Matrix4x4Array.size());
					break;
				case EmptySource::EMaterialDataType::FloatArray:
					SetFloat1Array(Uniform.Name.c_str(), &Uniform.FloatArray[0], (int)Uniform.FloatArray.size());
					break;
				case EmptySource::EMaterialDataType::Float2DArray:
					SetFloat2Array(Uniform.Name.c_str(), Uniform.Float2DArray[0].PointerToValue(), (int)Uniform.Float2DArray.size());
					break;
				case EmptySource::EMaterialDataType::Float3DArray:
					SetFloat3Array(Uniform.Name.c_str(), Uniform.Float3DArray[0].PointerToValue(), (int)Uniform.Float3DArray.size());
					break;
				case EmptySource::EMaterialDataType::Float4DArray:
					SetFloat4Array(Uniform.Name.c_str(), Uniform.Float4DArray[0].PointerToValue(), (int)Uniform.Float4DArray.size());
					break;
				case EmptySource::EMaterialDataType::Texture2D:
					SetTexture2D(Uniform.Name.c_str(), Uniform.Texture, i); i++;
					break;
				case EmptySource::EMaterialDataType::Cubemap:
					SetTextureCubemap(Uniform.Name.c_str(), Uniform.Texture, i); i++;
					break;
				case EmptySource::EMaterialDataType::IntArray:
					SetInt1Array(Uniform.Name.c_str(), &Uniform.IntArray[0], (int)Uniform.IntArray.size());
					break;
				case EmptySource::EMaterialDataType::None:
				default:
					break;
				}
			}
		}
	}

}