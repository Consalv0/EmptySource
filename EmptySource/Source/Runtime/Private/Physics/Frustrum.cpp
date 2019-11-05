#include "CoreMinimal.h"
#include "..\..\Public\Physics\Frustrum.h"

namespace ESource {

	Frustrum::Frustrum() {
		for (int i = 0; i < 6; ++i) {
			Planes[i] = Plane();
		}
	}

	Frustrum::Frustrum(const Frustrum & Other) {
		for (int i = 0; i < 6; ++i) {
			Planes[i] = Other.Planes[i];
		}
	}

	Frustrum Frustrum::FromProjectionViewMatrix(const Matrix4x4 & ComboMatrix) {
		Frustrum Result;
		Result.Planes[FP_Near  ].X = ComboMatrix[0][3] + ComboMatrix[0][2];
		Result.Planes[FP_Near  ].Y = ComboMatrix[1][3] + ComboMatrix[1][2];
		Result.Planes[FP_Near  ].Z = ComboMatrix[2][3] + ComboMatrix[2][2];
		Result.Planes[FP_Near  ].D = ComboMatrix[3][3] + ComboMatrix[3][2];
		Result.Planes[FP_Far   ].X = ComboMatrix[0][3] - ComboMatrix[0][2];
		Result.Planes[FP_Far   ].Y = ComboMatrix[1][3] - ComboMatrix[1][2];
		Result.Planes[FP_Far   ].Z = ComboMatrix[2][3] - ComboMatrix[2][2];
		Result.Planes[FP_Far   ].D = ComboMatrix[3][3] - ComboMatrix[3][2];
		Result.Planes[FP_Top   ].X = ComboMatrix[0][3] - ComboMatrix[0][1];
		Result.Planes[FP_Top   ].Y = ComboMatrix[1][3] - ComboMatrix[1][1];
		Result.Planes[FP_Top   ].Z = ComboMatrix[2][3] - ComboMatrix[2][1];
		Result.Planes[FP_Top   ].D = ComboMatrix[3][3] - ComboMatrix[3][1];
		Result.Planes[FP_Bottom].X = ComboMatrix[0][3] + ComboMatrix[0][1];
		Result.Planes[FP_Bottom].Y = ComboMatrix[1][3] + ComboMatrix[1][1];
		Result.Planes[FP_Bottom].Z = ComboMatrix[2][3] + ComboMatrix[2][1];
		Result.Planes[FP_Bottom].D = ComboMatrix[3][3] + ComboMatrix[3][1];
		Result.Planes[FP_Left  ].X = ComboMatrix[0][3] + ComboMatrix[0][0];
		Result.Planes[FP_Left  ].Y = ComboMatrix[1][3] + ComboMatrix[1][0];
		Result.Planes[FP_Left  ].Z = ComboMatrix[2][3] + ComboMatrix[2][0];
		Result.Planes[FP_Left  ].D = ComboMatrix[3][3] + ComboMatrix[3][0];
		Result.Planes[FP_Right ].X = ComboMatrix[0][3] - ComboMatrix[0][0];
		Result.Planes[FP_Right ].Y = ComboMatrix[1][3] - ComboMatrix[1][0];
		Result.Planes[FP_Right ].Z = ComboMatrix[2][3] - ComboMatrix[2][0];
		Result.Planes[FP_Right ].D = ComboMatrix[3][3] - ComboMatrix[3][0];
		for (int i = 0; i < 6; ++i) Result.Planes[i].Normalize();
		return Result;
	}

	// https://cgvr.cs.uni-bremen.de/teaching/cg_literatur/lighthouse3d_view_frustum_culling/index.html
	ECullingResult Frustrum::CheckAABox(const Box3D & AABox) {
		ECullingResult Result = ECullingResult::Inside;
		Vector3 aMin, aMax;
		Vector3 vMin, vMax;

		aMin = AABox.GetMinPoint();
		aMin = AABox.GetMaxPoint();

		for (int i = 0; i < 6; ++i) {
			// Is the positive vertex outside?
			if (Planes[i].SignedDistance(AABox.GetPointPositive(Planes[i].Normal)) < 0)
				return ECullingResult::Outside;
			// Is the negative vertex outside?	
			else if (Planes[i].SignedDistance(AABox.GetPointNegative(Planes[i].Normal)) < 0)
				Result = ECullingResult::Intersect;
		}
		return Result;
	}

}