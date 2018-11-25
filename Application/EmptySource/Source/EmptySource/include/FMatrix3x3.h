#pragma once

struct FVector3;
struct FMatrix4x4;

struct FMatrix3x3 {
public:
	struct {
		float
			m00, m01, m02,
			m10, m11, m12,
			m20, m21, m22;
	};

	FMatrix3x3();
	FMatrix3x3(const FMatrix4x4& Matrix);
	FMatrix3x3(const FVector3& Row0, const FVector3& Row1, const FVector3& Row2);

	static FMatrix3x3 Identity();

	void Transpose();
	FMatrix3x3 Transposed() const;

	FVector3 Row(const int& i) const;
	FVector3 Column(const int& i) const;
	FMatrix3x3 operator*(const FMatrix3x3& Other) const;
	FVector3 operator*(const FVector3& Vector) const;

	const float* PoiterToValue(void) const;
};