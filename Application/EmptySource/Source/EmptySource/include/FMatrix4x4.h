#pragma once

struct FVector4;

struct FMatrix4x4 {

public:
	struct {
		float
			m00, m01, m02, m03,
			m10, m11, m12, m13,
			m20, m21, m22, m23,
			m30, m31, m32, m33;
	};
	
	FMatrix4x4();
	FMatrix4x4(const FVector4& Row0, const FVector4& Row1, const FVector4& Row2, const FVector4 Row3);

	static FMatrix4x4 Identity();
	static FMatrix4x4 Perspective(const float& Aperture, const float& Aspect, const float& Near, const float& Far);
	static FMatrix4x4 LookAt(const FVector3& Position, const FVector3& Direction, const FVector3& Up);

	void Transpose();
	FMatrix4x4 Transposed() const;

	FVector4 Row(const int& i) const;
	FVector4 Column(const int& i) const;
	
	FMatrix4x4 operator*(const FMatrix4x4& Other) const;
	FVector4 operator*(const FVector4& Vector) const;
	FVector3 operator*(const FVector3& Vector) const;

	const float* PoiterToValue(void) const;
};