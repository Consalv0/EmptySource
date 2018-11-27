#pragma once

struct FVector4;

struct FMatrix4x4 {

public:
	struct {
		FVector4 m0, m1, m2, m3;
	};

	FMatrix4x4();
	FMatrix4x4(const FMatrix4x4& Other);
	FMatrix4x4(const FVector4& Row0, const FVector4& Row1, const FVector4& Row2, const FVector4 Row3);

	static FMatrix4x4 Identity();
	static FMatrix4x4 Perspective(const float& Aperture, const float& Aspect, const float& Near, const float& Far);
	static FMatrix4x4 LookAt(const FVector3& Position, const FVector3& Direction, const FVector3& Up);

	void Transpose();
	FMatrix4x4 Transposed() const;
	FMatrix4x4 Inversed() const;

	FVector4 Row(const int& i) const;
	FVector4 Column(const int& i) const;

	FVector4 & operator[](unsigned int i);
	FVector4 const& operator[](unsigned int i) const;
	const float* PointerToValue(void) const;
	
	FMatrix4x4 operator*(const FMatrix4x4& Other) const;
	FVector4 operator*(const FVector4& Vector) const;
	FVector3 operator*(const FVector3& Vector) const;
	FMatrix4x4 operator*(const float& Value) const;
	FMatrix4x4 operator/(const float& Value) const;
};