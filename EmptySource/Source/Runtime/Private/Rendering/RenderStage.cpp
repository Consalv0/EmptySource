
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/Window.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/RenderStage.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"

#include "Resources/ShaderManager.h"
#include "Resources/TextureManager.h"
#include "Rendering/MeshPrimitives.h"

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

	void RenderStage::SetRenderTarget(const RenderTargetPtr & InTarget) {
		Target = InTarget;
	}

	void RenderStage::End() {
		if (Target == NULL) return;
		Scene.RenderLightMap(0, ShaderManager::GetInstance().GetProgram(L"DepthTestShader"));
		Scene.RenderLightMap(1, ShaderManager::GetInstance().GetProgram(L"DepthTestShader")); 
		Rendering::SetViewport({ 0.F, 0.F, (float)Target->GetSize().x, (float)Target->GetSize().y });
		Target->Bind();
		Target->Clear();
		Scene.Render();
		Target->Unbind();

		RShaderPtr BloomThresholdShader = ShaderManager::GetInstance().GetProgram(L"PostProcessingBloomThreshold");
		RShaderPtr BloomShader = ShaderManager::GetInstance().GetProgram(L"PostProcessingBloom");
		RShaderPtr RenderShader = ShaderManager::GetInstance().GetProgram(L"RenderScreenTexture");
		RTexturePtr BloomThresholdTexture = TextureManager::GetInstance().CreateTexture2D(L"PPBloomThreshold", L"", PF_RGB16F, FM_MinMagLinear, SAM_Clamp, 0); 
		if (BloomThresholdTexture) {
			if (BloomThresholdTexture->GetSize() != Target->GetSize() / 2) {
				BloomThresholdTexture->Unload();
				BloomThresholdTexture->SetSize(Target->GetSize() / 2);
				BloomThresholdTexture->Load();
			}
		}
		RTexturePtr BloomHorizontalTexture = TextureManager::GetInstance().CreateTexture2D(L"PPBloomHorizontalPass", L"", PF_RGB16F, FM_MinMagLinear, SAM_Clamp, 0);
		if (BloomHorizontalTexture) {
			if (BloomHorizontalTexture->GetSize() != Target->GetSize() / 2) {
				BloomHorizontalTexture->Unload();
				BloomHorizontalTexture->SetSize(Target->GetSize() / 2);
				BloomHorizontalTexture->Load();
			}
		}
		RTexturePtr BloomTexture = TextureManager::GetInstance().CreateTexture2D(L"PPBloom", L"", PF_RGB16F, FM_MinMagLinear, SAM_Clamp, 0);
		if (BloomTexture) {
			if (BloomTexture->GetSize() != Target->GetSize() / 2) {
				BloomTexture->Unload();
				BloomTexture->SetSize(Target->GetSize() / 2);
				BloomTexture->Load();
			}
		}

		Rendering::SetViewport({ 0.F, 0.F, (float)(Application::GetInstance()->GetWindow().GetWidth() / 2), (float)(Application::GetInstance()->GetWindow().GetHeight() / 2) });

		RenderTargetPtr BloomThresholdTarget = RenderTarget::Create();
		BloomThresholdTarget->BindTexture2D((Texture2D *)BloomThresholdTexture->GetNativeTexture(), BloomThresholdTexture->GetSize());
		BloomThresholdTarget->Bind();
		if (BloomThresholdShader && BloomThresholdShader->IsValid()) {
			BloomThresholdShader->GetProgram()->Bind();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);
			BloomThresholdShader->GetProgram()->SetTexture("_MainTexture", Target->GetBindedTexture(0), 0);
			const float Treshold = 1.0F;
			BloomThresholdShader->GetProgram()->SetFloat1Array("_Threshold", &Treshold);
			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			BloomThresholdShader->GetProgram()->Unbind();
		}
		BloomThresholdTarget->Unbind();
		BloomThresholdTexture->DeleteMipMaps();
		BloomThresholdTexture->GenerateMipMaps();

		if (BloomShader && BloomShader->IsValid()) {
			BloomShader->GetProgram()->Bind();
		
			Vector2 Radius(10.0F, 10.0F);
			Vector2 HorizontalDirection(1.0F, 0.0F);
			Vector2 VerticalDirection(0.0F, 1.0F);
		
			RenderTargetPtr BloomHorizontalBlurTarget = RenderTarget::Create();
			BloomHorizontalBlurTarget->BindTexture2D((Texture2D *)BloomHorizontalTexture->GetNativeTexture(), BloomHorizontalTexture->GetSize());
			BloomHorizontalBlurTarget->Bind();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);
			BloomShader->GetProgram()->SetTexture("_MainTexture", BloomThresholdTexture->GetNativeTexture(), 0);
			BloomShader->GetProgram()->SetFloat1Array("_Radius", &Radius[0], 1);
			BloomShader->GetProgram()->SetFloat2Array("_Direction", HorizontalDirection.PointerToValue(), 1);
			BloomShader->GetProgram()->SetFloat2Array("_Resolution", Vector2(BloomThresholdTexture->GetSize()).PointerToValue(), 1);
			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			BloomHorizontalBlurTarget->Unbind();
			BloomHorizontalTexture->DeleteMipMaps();
			BloomHorizontalTexture->GenerateMipMaps();
		
			RenderTargetPtr BloomBlurTarget = RenderTarget::Create();
			BloomBlurTarget->BindTexture2D((Texture2D *)BloomTexture->GetNativeTexture(), BloomTexture->GetSize());
			BloomBlurTarget->Bind();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);
			BloomShader->GetProgram()->SetTexture("_MainTexture", BloomHorizontalTexture->GetNativeTexture(), 0);
			BloomShader->GetProgram()->SetFloat1Array("_Radius", &Radius[1], 1);
			BloomShader->GetProgram()->SetFloat2Array("_Direction", VerticalDirection.PointerToValue(), 1);
			BloomShader->GetProgram()->SetFloat2Array("_Resolution", Vector2(BloomHorizontalTexture->GetSize()).PointerToValue(), 1);
			
			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			BloomBlurTarget->Unbind();
			BloomShader->GetProgram()->Unbind();
		}

		Rendering::SetViewport({ 0.F, 0.F, (float)Application::GetInstance()->GetWindow().GetWidth(), (float)Application::GetInstance()->GetWindow().GetHeight() });

		if (RenderShader && RenderShader->IsValid()) {
			RenderShader->GetProgram()->Bind();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);

			RenderShader->GetProgram()->SetTexture("_MainTexture", Target->GetBindedTexture(0), 0);
			RenderShader->GetProgram()->SetTexture("_BloomTexture", BloomTexture->GetNativeTexture(), 1);
			RenderShader->GetProgram()->SetFloat1Array("_Exposure", &Pipeline->Exposure, 1);
			RenderShader->GetProgram()->SetFloat1Array("_Gamma", &Pipeline->Gamma, 1);

			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			RenderShader->GetProgram()->Unbind();
		}


	}

	void RenderStage::Begin() {
		Scene.Clear();
	}

}