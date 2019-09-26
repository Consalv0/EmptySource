#pragma once

#include "Resources/ResourceManager.h"
#include "ResourceHolder.h"
#include "Resources/ShaderParameters.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	typedef std::shared_ptr<class RShader> RShaderPtr;

	class RShader : public ResourceHolder {
	public:
		~RShader();

		virtual bool IsValid() const override;

		virtual void Load() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }
		
		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Shader; }

		virtual inline size_t GetMemorySize() const override { return sizeof(unsigned int); };

		inline ShaderProgram * GetProgram() { return ShaderPointer; };

		static inline EResourceType GetType() { return EResourceType::RT_Shader; };

		//* Get the chader properties
		const TArray<ShaderParameters> & GetParameters() const { return Parameters; }

		//* Set the shader properties
		void SetParameters(const TArrayInitializer<ShaderParameters> & Parameters);

		//* Set the shader properties
		void SetParameters(const TArray<ShaderParameters> & Parameters);

		inline const NString& GetSourceCode() const { return SourceCode; }

	protected:
		friend class ShaderManager;

		RShader(const IName & Name, const WString & Origin, const NString& Source);

		bool LoadFromShaderSource(const NString & Source);

	private:
		ShaderProgram * ShaderPointer;
		TArray<ShaderStage *> Stages;

		NString SourceCode;

		TArray<ShaderParameters> Parameters;
	};

}
