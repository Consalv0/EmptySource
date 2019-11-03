#pragma once

namespace ESource {

	enum FrustrumPlanes {
		FrustrumPlanes_Near   = 0,
		FrustrumPlanes_Far    = 1,
		FrustrumPlanes_Top    = 2,
		FrustrumPlanes_Bottom = 3,
		FrustrumPlanes_Left   = 4,
		FrustrumPlanes_Right  = 5
	};

	enum class CullingResult {
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
		
		CullingResult CheckAABox(const Box3D & AABox);

	};

}