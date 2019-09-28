
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/Window.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/RenderStage.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"

#include "Resources/ShaderManager.h"
#include "Resources/TextureManager.h"

#include "Utility/TextFormattingMath.h"

namespace ESource {
	
	RenderStage::RenderStage(const IName & Name, RenderPipeline * Pipeline) 
		: Name(Name), Scene(), Pipeline(Pipeline) {
	}

	void RenderStage::SubmitVertexArray(const VertexArrayPtr & VertexArray, const MaterialPtr & Mat, const Matrix4x4 & Matrix) {
		if (Mat->GetShaderProgram() == NULL || Mat->GetShaderProgram()->GetLoadState() != LS_Loaded) return;
		Scene.Submit(Mat, VertexArray, Matrix);
	}

	void RenderStage::SubmitPointLight(unsigned int Index, const Transform & Transformation, const Vector3 & Color, const float & Intensity) {
		Scene.Lights[Index].Transformation = Transformation;
		Scene.Lights[Index].Color = Color;
		Scene.Lights[Index].Intensity = Intensity;
	}

	void RenderStage::SubmitDirectionalLight(unsigned int Index, const Transform & Transformation, const Vector3 & Color, const float & Intensity, const Matrix4x4 & Projection) {
		Scene.Lights[Index].Transformation = Transformation;
		Scene.Lights[Index].Color = Color;
		Scene.Lights[Index].Intensity = Intensity;
		Scene.Lights[Index].ProjectionMatrix = Projection;
		Scene.Lights[Index].ShadowMap = NULL;
	}

	void RenderStage::SubmitDirectionalShadowMap(unsigned int Index, const RTexturePtr & Texture, const float & Bias) {
		Scene.Lights[Index].ShadowMap = Texture;
		Scene.Lights[Index].ShadowBias = Bias;
	}

	void RenderStage::SetEyeTransform(const Transform & EyeTransform) {
		Scene.EyeTransform = EyeTransform;
	}

	void RenderStage::SetProjectionMatrix(const Matrix4x4 & Projection) {
		Scene.ViewProjection = Projection;
	}

	void RenderStage::End() {
		Scene.RenderLightMap(0, ShaderManager::GetInstance().GetProgram(L"DepthTestShader"));
		Scene.Render();
	}

	void RenderStage::Begin() {
		Scene.Clear();
	}

}