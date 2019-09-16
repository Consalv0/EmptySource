
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/Window.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/RenderStage.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"

#include "Resources/TextureManager.h"

#include "Utility/TextFormattingMath.h"

namespace EmptySource {

	void RenderStage::SubmitMesh(const MeshPtr & Model, int Subdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix) {
		Model->BindSubdivisionVertexArray(Subdivision);

		float CubemapTextureMipmaps = (float)TextureManager::GetInstance().GetTexture(L"CubemapTexture")->GetMipMapCount();
		Mat->SetVariables({
			{ "_ViewPosition",        { EyeTransform.Position }, SPFlags_IsInternal },
			{ "_ProjectionMatrix",    { ViewProjection }, SPFlags_IsInternal },
			{ "_ViewMatrix",          { EyeTransform.GetGLViewMatrix() }, SPFlags_IsInternal },
			{ "_Lights[0].Position",  { SceneLights[0].Position }, SPFlags_IsInternal },
			{ "_Lights[0].Color",     { SceneLights[0].Color }, SPFlags_IsInternal | SPFlags_IsColor },
			{ "_Lights[0].Intencity", { SceneLights[0].Intensity }, SPFlags_IsInternal },
			{ "_Lights[1].Position",  { SceneLights[1].Position }, SPFlags_IsInternal },
			{ "_Lights[1].Color",     { SceneLights[1].Color }, SPFlags_IsInternal | SPFlags_IsColor },
			{ "_Lights[1].Intencity", { SceneLights[1].Intensity }, SPFlags_IsInternal },
			{ "_GlobalTime",          { Time::GetEpochTime<Time::Second>() }, SPFlags_IsInternal },
			{ "_BRDFLUT",             { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"BRDFLut") }, SPFlags_IsInternal },
			{ "_EnviromentMap",       { ETextureDimension::Cubemap, TextureManager::GetInstance().GetTexture(L"CubemapTexture") }, SPFlags_IsInternal },
			{ "_EnviromentMapLods",   { CubemapTextureMipmaps }, SPFlags_IsInternal }
		});
		Mat->Use();
		Mat->SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix.PointerToValue(), GetMatrixBuffer());
		Model->DrawSubdivisionInstanciated(1, Subdivision);
	}

	void RenderStage::SetLight(unsigned int Index, const Point3 & Position, const Vector3 & Color, const float & Intensity) {
		SceneLights[Index].Position = Position;
		SceneLights[Index].Color = Color;
		SceneLights[Index].Intensity = Intensity;
	}

	void RenderStage::SetEyeTransform(const Transform & EyeTransform) {
		this->EyeTransform = EyeTransform;
	}

	void RenderStage::SetProjectionMatrix(const Matrix4x4 & Projection) {
		ViewProjection = Projection;
	}

	VertexBufferPtr RenderStage::GetMatrixBuffer() const {
		return ModelMatrixBuffer;
	}

	void RenderStage::Initialize() {
		ModelMatrixBuffer = VertexBuffer::Create(NULL, 0, EUsageMode::UM_Dynamic);
	}

}