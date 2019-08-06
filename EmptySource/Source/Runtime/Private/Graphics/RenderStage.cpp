
#include "Engine/Application.h"
#include "Engine/Window.h"
#include "Graphics/RenderPipeline.h"
#include "Graphics/RenderStage.h"
#include "Graphics/Material.h"
#include "Graphics/GLFunctions.h"
#include "Graphics/Cubemap.h"
#include "Resources/Resources.h"
#include "Mesh/Mesh.h"
#include "Utility/TextFormattingMath.h"

namespace EmptySource {

	void RenderStage::SetEyeTransform(const Transform & EyeTransform) {
		this->EyeTransform = EyeTransform;
	}

	void RenderStage::SetViewProjection(const Matrix4x4 & Projection) {
		ViewProjection = Projection;
	}

	unsigned int RenderStage::GetMatrixBuffer() const {
		return ModelMatrixBuffer;
	}

	void RenderStage::Initialize() {
		glGenBuffers(1, &ModelMatrixBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
	}

	void RenderStage::Prepare() {
		if (CurrentMaterial == NULL) return;

		CurrentMaterial->Use();

		float Value0 = 1.0F;
		float Value1 = 0.0F;

		CurrentMaterial->SetFloat3Array("_ViewPosition", EyeTransform.Position.PointerToValue());
		CurrentMaterial->SetFloat3Array("_Lights[0].Position", Vector3(0).PointerToValue());
		CurrentMaterial->SetFloat3Array("_Lights[0].Color", Vector3(0, 0, 1).PointerToValue());
		CurrentMaterial->SetFloat1Array("_Lights[0].Intencity", &Value1);
		CurrentMaterial->SetFloat3Array("_Lights[1].Position", Vector3(0, 5, -5).PointerToValue());
		CurrentMaterial->SetFloat3Array("_Lights[1].Color", Vector3(1, 1, 1).PointerToValue());
		CurrentMaterial->SetFloat1Array("_Lights[1].Intencity", &Value1);
		CurrentMaterial->SetFloat1Array("_Material.Metalness", &Value0);
		CurrentMaterial->SetFloat1Array("_Material.Roughness", &Value0);
		CurrentMaterial->SetFloat3Array("_Material.Color", Vector3(1.F).PointerToValue());

		CurrentMaterial->SetMatrix4x4Array("_ProjectionMatrix", ViewProjection.PointerToValue());
		CurrentMaterial->SetMatrix4x4Array("_ViewMatrix", EyeTransform.GetGLViewMatrix().PointerToValue());
		CurrentMaterial->SetTexture2D("_MainTexture", OldResourceManager::Get<Texture2D>(L"FlamerAlbedoTexture")->GetData(), 0);
		CurrentMaterial->SetTexture2D("_NormalTexture", OldResourceManager::Get<Texture2D>(L"FlamerNormalTexture")->GetData(), 1);
		CurrentMaterial->SetTexture2D("_RoughnessTexture", OldResourceManager::Get<Texture2D>(L"FlamerRoughnessTexture")->GetData(), 2);
		CurrentMaterial->SetTexture2D("_MetallicTexture", OldResourceManager::Get<Texture2D>(L"FlamerMetallicTexture")->GetData(), 3);
		CurrentMaterial->SetTexture2D("_BRDFLUT", OldResourceManager::Get<Texture2D>(L"BRDFLut")->GetData(), 4);
		CurrentMaterial->SetTextureCubemap("_EnviromentMap", OldResourceManager::Get<Cubemap>(L"CubemapTexture")->GetData(), 5);
		float CubemapTextureMipmaps = OldResourceManager::Get<Cubemap>(L"CubemapTexture")->GetData()->GetMipmapCount();
		CurrentMaterial->SetFloat1Array("_EnviromentMapLods", &CubemapTextureMipmaps);
	}

	void RenderStage::RunStage() {
		Prepare();
		OnRenderEvent.Notify();
		Finish();
	}

}