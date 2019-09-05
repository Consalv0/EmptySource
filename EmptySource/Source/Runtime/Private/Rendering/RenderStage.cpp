
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/Window.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/RenderStage.h"
#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"
#include "Utility/TextFormattingMath.h"

namespace EmptySource {

	void RenderStage::SubmitMesh(const MeshPtr & Model, int Subdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix) {
		Model->BindSubdivisionVertexArray(Subdivision);

		Mat->SetVariables({
			{ "_ViewPosition", TArrayInitializer<Vector3>({ EyeTransform.Position }) },
			{ "_ProjectionMatrix", { ViewProjection } },
			{ "_ViewMatrix", { EyeTransform.GetGLViewMatrix() } },
			{ "_Lights[0].Position", TArrayInitializer<Vector3>({  Vector3(2, 1) }) },
			{ "_Lights[0].Color", TArrayInitializer<Vector3>({ Vector3(1.F, 1.F, .9F) }) },
			{ "_Lights[0].Intencity", TArrayInitializer<float>({ 20 }) },
			{ "_Lights[1].Position", TArrayInitializer<Vector3>({  Vector3(-2, 1) }) },
			{ "_Lights[1].Color", TArrayInitializer<Vector3>({ Vector3(1.F, 1.F, .9F) }) },
			{ "_Lights[1].Intencity", TArrayInitializer<float>({ 20 }) },
		});
		Mat->Use();
		Mat->SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix.PointerToValue(), GetMatrixBuffer());
		Model->DrawSubdivisionInstanciated(1, Subdivision);
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