
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

	void RenderStage::SubmitPointLight(const Transform & Transformation, const Vector3 & Color, const float & Intensity) {
		Scene.LightCount++;
		Scene.Lights[Scene.LightCount].Transformation = Transformation;
		Scene.Lights[Scene.LightCount].Color = Color;
		Scene.Lights[Scene.LightCount].Direction = 0.F;
		Scene.Lights[Scene.LightCount].Intensity = Intensity;
		Scene.Lights[Scene.LightCount].CastShadow = false;
		Scene.Lights[Scene.LightCount].ShadowMap = NULL;
	}

	void RenderStage::SubmitSpotLight(const Transform & Transformation, const Vector3 & Color, const Vector3& Direction, const float & Intensity, const Matrix4x4 & Projection) {
		Scene.LightCount++;
		Scene.Lights[Scene.LightCount].Transformation = Transformation;
		Scene.Lights[Scene.LightCount].Color = Color;
		Scene.Lights[Scene.LightCount].Direction = Direction;
		Scene.Lights[Scene.LightCount].Intensity = Intensity;
		Scene.Lights[Scene.LightCount].ProjectionMatrix = Projection;
		Scene.Lights[Scene.LightCount].ShadowMap = NULL;
		Scene.Lights[Scene.LightCount].CastShadow = false;
	}

	void RenderStage::SubmitSpotShadowMap(const RTexturePtr & Texture, const float & Bias) {
		if (Scene.LightCount < 0) return;
		Scene.Lights[Scene.LightCount].ShadowMap = Texture;
		Scene.Lights[Scene.LightCount].ShadowBias = Bias;
		Scene.Lights[Scene.LightCount].CastShadow = true;
	}

	void RenderStage::SetEyeTransform(const Transform & EyeTransform) {
		Scene.EyeTransform = EyeTransform;
	}

	void RenderStage::SetProjectionMatrix(const Matrix4x4 & Projection) {
		Scene.ViewProjection = Projection;
	}

	void RenderStage::End() {
		Scene.RenderLightMap(0, ShaderManager::GetInstance().GetProgram(L"DepthTestShader"));
		Scene.RenderLightMap(1, ShaderManager::GetInstance().GetProgram(L"DepthTestShader"));
		Scene.Render();
	}

	void RenderStage::Begin() {
		Scene.Clear();
	}

}