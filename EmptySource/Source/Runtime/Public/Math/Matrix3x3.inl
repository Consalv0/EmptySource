
#include <math.h>
#include <stdexcept>

#include "Math/Vector3.h"
#include "Math/Matrix4x4.h"
#include "Math/Matrix3x3.h"

namespace ESource {

	FORCEINLINE Matrix3x3::Matrix3x3() {
		m0[0] = 1; m0[1] = 0; m0[2] = 0;
		m1[0] = 0; m1[1] = 1; m1[2] = 0;
		m2[0] = 0; m2[1] = 0; m2[2] = 1;
	}

	FORCEINLINE Matrix3x3::Matrix3x3(const Matrix3x3 & Matrix)
		: m0(Matrix.m0), m1(Matrix.m1), m2(Matrix.m2) {
	}

	FORCEINLINE Matrix3x3::Matrix3x3(const Matrix4x4 & Matrix) {
		Matrix3x3(Matrix.GetRow(0), Matrix.GetRow(1), Matrix.GetRow(2));
	}

	FORCEINLINE Matrix3x3::Matrix3x3(const Vector3 & Row0, const Vector3 & Row1, const Vector3 & Row2) {
		m0[0] = Row0.X; m0[1] = Row0.Y; m0[2] = Row0.Z;
		m1[0] = Row1.X; m1[1] = Row1.Y; m1[2] = Row1.Z;
		m2[0] = Row2.X; m2[1] = Row2.Y; m2[2] = Row2.Z;
	}

	inline Matrix3x3 Matrix3x3::Identity() {
		return Matrix3x3();
	}

	inline void Matrix3x3::Transpose() {
		*this = Matrix3x3(GetColumn(0), GetColumn(1), GetColumn(2));
	}

	inline Matrix3x3 Matrix3x3::Transposed() const {
		return Matrix3x3(GetColumn(0), GetColumn(1), GetColumn(2));
	}

	inline Vector3 Matrix3x3::GetRow(const unsigned char & i) const {
		if ((i > 2)) return Vector3();
		return ((Vector3*)this)[i];
	}

	inline Vector3 Matrix3x3::GetColumn(const unsigned char & i) const {
		switch (i) {
			case 0: return Vector3(m0[0], m1[0], m2[0]);
			case 1: return Vector3(m0[1], m1[1], m2[1]);
			case 2: return Vector3(m0[2], m1[2], m2[2]);
		}

		return Vector3();
	}

	inline Vector3 & Matrix3x3::operator[](unsigned char i) {
		ES_CORE_ASSERT(i <= 2, "Matrix3x3 index out of bounds");
		return ((Vector3*)this)[i];
	}

	inline Vector3 const & Matrix3x3::operator[](unsigned char i) const {
		ES_CORE_ASSERT(i <= 2, "Matrix3x3 index out of bounds");
		return ((Vector3*)this)[i];
	}

	FORCEINLINE Matrix3x3 Matrix3x3::operator*(const Matrix3x3& Other) const {
		Matrix3x3 Result = Matrix3x3();
		const Vector3 Col0 = GetColumn(0), Col1 = GetColumn(1), Col2 = GetColumn(2);

		Result.m0[0] = Other.m0.Dot(Col0);
		Result.m1[0] = Other.m1.Dot(Col0);
		Result.m2[0] = Other.m2.Dot(Col0);

		Result.m0[1] = Other.m0.Dot(Col1);
		Result.m1[1] = Other.m1.Dot(Col1);
		Result.m2[1] = Other.m2.Dot(Col1);

		Result.m0[2] = Other.m0.Dot(Col2);
		Result.m1[2] = Other.m1.Dot(Col2);
		Result.m2[2] = Other.m2.Dot(Col2);

		return Result;
	}

	FORCEINLINE Vector3 Matrix3x3::operator*(const Vector3& Vector) const {
		Vector3 result(
			GetColumn(0).Dot(Vector),
			GetColumn(1).Dot(Vector),
			GetColumn(2).Dot(Vector)
		);

		return result;
	}

	inline const float * Matrix3x3::PointerToValue(void) const {
		return &m0[0];
	}

}