#pragma once

struct FVector3;
struct FMatrix4x4;

struct FMatrix3x3 {
public:
	struct {
		FVector3 m0, m1, m2;
	};

	FMatrix3x3();
	FMatrix3x3(const FMatrix3x3& Matrix);
	FMatrix3x3(const FMatrix4x4& Matrix);
	FMatrix3x3(const FVector3& Row0, const FVector3& Row1, const FVector3& Row2);

	static FMatrix3x3 Identity();

	void Transpose();
	FMatrix3x3 Transposed() const;

	FVector3 Row(const int& i) const;
	FVector3 Column(const int& i) const;

	FVector3 & operator[](unsigned int i);
	FVector3 const& operator[](unsigned int i) const;
	const float* PoiterToValue(void) const;

	FMatrix3x3 operator*(const FMatrix3x3& Other) const;
	FVector3 operator*(const FVector3& Vector) const;
};