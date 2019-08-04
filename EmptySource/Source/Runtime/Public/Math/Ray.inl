
#include "Math/Ray.h"

namespace EmptySource {

	FORCEINLINE Ray::Ray() {
		Origin = Direction = Vector3();
	}

	FORCEINLINE Ray::Ray(const Vector3 & Origin, const Vector3 & Direction)
		: Origin(Origin), Direction(Direction)
	{ }

	inline Vector3 Ray::PointAt(float t) const {
		return Origin + (Direction * t);
	}

}