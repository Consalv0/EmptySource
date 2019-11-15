#pragma once

#include "Resources/ResourceManager.h"
#include "ResourceHolder.h"
#include "Resources/ShaderParameters.h"
#include "Rendering/Shader.h"

namespace ESource {

	typedef std::shared_ptr<class RShader> RShaderPtr;

	class RShader : public ResourceHolder {
	public:
		~RShader();

		virtual bool IsValid() const override;

		virtual void Load() override;

		virtual void LoadAsync() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }
		
		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Shader; }

		virtual inline size_t GetMemorySize() const override { return sizeof(uint32_t); };

		inline ShaderProgram * GetProgram() { return ShaderPointer; };

		static inline EResourceType GetType() { return EResourceType::RT_Shader; };

		//* Get the chader properties
		const TArray<ShaderParameter> & GetParameters() const { return Parameters; }

		//* Set the shader properties
		void SetParameters(const TArrayInitializer<ShaderParameter> & Parameters);

		//* Set the shader properties
		void SetParameters(const TArray<ShaderParameter> & Parameters);

		bool IsInstancing() { return CompileFlags & (int)EShaderCompileFalgs::Instancing; }

		inline const NString& GetSourceCode() const { return SourceCode; }

	protected:
		friend class ShaderManager;

		RShader(const IName & Name, const WString & Origin, const NString& Source, int CompileFlags);

		bool LoadFromShaderSource(const NString & Source);

	private:
		ShaderProgram * ShaderPointer;
		TArray<ShaderStage *> Stages;

		NString SourceCode;
		int CompileFlags;

		TArray<ShaderParameter> Parameters;
	};

}
