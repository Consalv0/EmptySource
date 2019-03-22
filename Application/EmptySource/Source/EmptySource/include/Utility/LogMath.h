#pragma once

#include "../Math/CoreMath.h"
#include "../Text.h"

namespace Text {
	inline WString FormatMath(const IntVector2& Value) {
		return Formatted(L"{%d, %d}", Value.x, Value.y);
	}

	inline WString FormatMath(const IntVector3& Value) {
		return Formatted(L"{%d, %d, %d}", Value.x, Value.y, Value.z);
	}

	inline WString FormatMath(const Quaternion& Value) {
		return Formatted(L"{%.2f, %.2f, %.2f, %.2f}", Value.w, Value.x, Value.y, Value.z);
	}

	inline WString FormatMath(const Vector2& Value) {
		return Formatted(L"{%.2f, %.2f}", Value.x, Value.y);
	}

	inline WString FormatMath(const Vector3& Value) {
		return Formatted(L"{%.2f, %.2f, %.2f}", Value.x, Value.y, Value.z);
	}

	inline WString FormatMath(const Vector4& Value) {
		return Formatted(L"{%.2f, %.2f, %.2f, %.2f}", Value.x, Value.y, Value.z, Value.w);
	}
}
