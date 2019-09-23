
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
	
	RenderStage::RenderStage(const IName & Name, RenderPipeline * Pipeline) 
		: Name(Name), Scene(), Pipeline(Pipeline) {
	}

	void RenderStage::SubmitVertexArray(const VertexArrayPtr & VertexArray, const MaterialPtr & Mat, const Matrix4x4 & Matrix) {
		if (Mat->GetShaderProgram() == NULL || Mat->GetShaderProgram()->GetLoadState() != LS_Loaded) return;
		Scene.Submit(Mat, VertexArray, Matrix);
	}

	void RenderStage::SubmitLight(unsigned int Index, const Point3 & Position, const Vector3 & Color, const float & Intensity) {
		Scene.Lights[Index].Position = Position;
		Scene.Lights[Index].Color = Color;
		Scene.Lights[Index].Intensity = Intensity;
	}

	void RenderStage::SetEyeTransform(const Transform & EyeTransform) {
		Scene.EyeTransform = EyeTransform;
	}

	void RenderStage::SetProjectionMatrix(const Matrix4x4 & Projection) {
		Scene.ViewProjection = Projection;
	}

	void RenderStage::End() {
		Scene.Render();
	}

	void RenderStage::Begin() {
		Scene.Clear();
	}

}