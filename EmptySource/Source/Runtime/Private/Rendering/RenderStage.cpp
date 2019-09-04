
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

	void RenderStage::SetEyeTransform(const Transform & EyeTransform) {
		this->EyeTransform = EyeTransform;
	}

	void RenderStage::SetViewProjection(const Matrix4x4 & Projection) {
		ViewProjection = Projection;
	}

	VertexBufferPtr RenderStage::GetMatrixBuffer() const {
		return ModelMatrixBuffer;
	}

	void RenderStage::Initialize() {
		ModelMatrixBuffer = VertexBuffer::Create(NULL, 0, EUsageMode::UM_Dynamic);
	}

}