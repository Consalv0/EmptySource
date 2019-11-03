#pragma once

#include "Math/CoreMath.h"
#include "Utility/TextFormatting.h"

namespace ESource {

	namespace Text {
		inline WString FormatMath(const IntVector2& Value) {
			return Formatted(L"{%d, %d}", Value.X, Value.Y);
		}

		inline WString FormatMath(const IntVector3& Value) {
			return Formatted(L"{%d, %d, %d}", Value.X, Value.Y, Value.Z);
		}

		inline WString FormatMath(const Quaternion& Value) {
			return Formatted(L"{%.3f, %.3f, %.3f, %.3f}", Value.W, Value.X, Value.Y, Value.Z);
		}

		inline WString FormatMath(const Vector2& Value) {
			return Formatted(L"{%.3f, %.3f}", Value.X, Value.Y);
		}

		inline WString FormatMath(const Vector3& Value) {
			return Formatted(L"{%.3f, %.3f, %.3f}", Value.X, Value.Y, Value.Z);
		}

		inline WString FormatMath(const Vector4& Value) {
			return Formatted(L"{%.3f, %.3f, %.3f, %.3f}", Value.X, Value.Y, Value.Z, Value.W);
		}

		inline WString FormatMath(const Matrix4x4& Value, bool ColumnMajor = false) {
			if (ColumnMajor) {
				return Formatted(L"{{%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}}",
					Value.m0.X, Value.m1.X, Value.m2.X, Value.m3.X,
					Value.m0.Y, Value.m1.Y, Value.m2.Y, Value.m3.Y,
					Value.m0.Z, Value.m1.Z, Value.m2.Z, Value.m3.Z,
					Value.m0.W, Value.m1.W, Value.m2.W, Value.m3.W
				);
			}
			else {
				return Formatted(L"{{%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f, %.3f}}",
					Value.m0.X, Value.m0.Y, Value.m0.Z, Value.m0.W,
					Value.m1.X, Value.m1.Y, Value.m1.Z, Value.m1.W,
					Value.m2.X, Value.m2.Y, Value.m2.Z, Value.m2.W,
					Value.m3.X, Value.m3.Y, Value.m3.Z, Value.m3.W
				);
			}
		}
	}

}