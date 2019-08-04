#pragma once

#include "Math/CoreMath.h"
#include "Engine/Text.h"

namespace EmptySource {

	namespace Text {
		inline WString FormatMath(const IntVector2& Value) {
			return Formatted(L"{%d, %d}", Value.x, Value.y);
		}

		inline WString FormatMath(const IntVector3& Value) {
			return Formatted(L"{%d, %d, %d}", Value.x, Value.y, Value.z);
		}

		inline WString FormatMath(const Quaternion& Value) {
			return Formatted(L"{%.3f, %.3f, %.3f, %.3f}", Value.w, Value.x, Value.y, Value.z);
		}

		inline WString FormatMath(const Vector2& Value) {
			return Formatted(L"{%.3f, %.3f}", Value.x, Value.y);
		}

		inline WString FormatMath(const Vector3& Value) {
			return Formatted(L"{%.3f, %.3f, %.3f}", Value.x, Value.y, Value.z);
		}

		inline WString FormatMath(const Vector4& Value) {
			return Formatted(L"{%.3f, %.3f, %.3f, %.3f}", Value.x, Value.y, Value.z, Value.w);
		}

		inline WString FormatMath(const Matrix4x4& Value, bool ColumnMajor = false) {
			if (ColumnMajor) {
				return Formatted(L"{{%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}}",
					Value.m0.x, Value.m1.x, Value.m2.x, Value.m3.x,
					Value.m0.y, Value.m1.y, Value.m2.y, Value.m3.y,
					Value.m0.z, Value.m1.z, Value.m2.z, Value.m3.z,
					Value.m0.w, Value.m1.w, Value.m2.w, Value.m3.w
				);
			}
			else {
				return Formatted(L"{{%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}}",
					Value.m0.x, Value.m0.y, Value.m0.z, Value.m0.w,
					Value.m1.x, Value.m1.y, Value.m1.z, Value.m1.w,
					Value.m2.x, Value.m2.y, Value.m2.z, Value.m2.w,
					Value.m3.x, Value.m3.y, Value.m3.z, Value.m3.w
				);
			}
		}
	}

}