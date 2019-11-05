#pragma once

namespace ESource {

	enum EFrustrumPlanes {
		FP_Near   = 0,
		FP_Far    = 1,
		FP_Top    = 2,
		FP_Bottom = 3,
		FP_Left   = 4,
		FP_Right  = 5
	};

	enum class ECullingResult {
		Outside,
		Inside,
		Intersect
	};

	class Frustrum {
	public:
		Plane Planes[6];

		Frustrum();

		Frustrum(const Frustrum & Other);

		static Frustrum FromProjectionViewMatrix(const Matrix4x4 & ComboMatrix);
		
		ECullingResult CheckAABox(const Box3D & AABox);

	};

}