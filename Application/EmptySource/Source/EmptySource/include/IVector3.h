#pragma once

struct FVector2;
struct FVector3;
struct FVector4;

struct IVector3 {
public:
	union {
		struct { int x, y, z; };
		struct { int r, g, b; };
	};

	IVector3();
	IVector3(const IVector3& Vector);
	IVector3(const FVector2& Vector);
	IVector3(const FVector3& Vector);
	IVector3(const FVector4& Vector);
	IVector3(const int& Value);
	IVector3(const int& x, const int& y, const int& z);
	IVector3(const int& x, const int& y);

	float Magnitude() const;
	int MagnitudeSquared() const;

	IVector3 Cross(const IVector3& Other) const;
	int Dot(const IVector3& Other) const;

	FVector3 FloatVector3() const;
	int & operator[](unsigned int i);
	int const& operator[](unsigned int i) const;
	const int* PoiterToValue() const;

	bool operator==(const IVector3& Other);
	bool operator!=(const IVector3& Other);

	IVector3 operator+(const IVector3& Other) const;
	IVector3 operator-(const IVector3& Other) const;
	IVector3 operator-(void) const;
	IVector3 operator*(const IVector3& Other) const;
	IVector3 operator/(const IVector3& Other) const;
	IVector3 operator*(const int& Value) const;
	IVector3 operator/(const int& Value) const;

	IVector3& operator+=(const IVector3& Other);
	IVector3& operator-=(const IVector3& Other);
	IVector3& operator*=(const IVector3& Other);
	IVector3& operator/=(const IVector3& Other);
	IVector3& operator*=(const int& Value);
	IVector3& operator/=(const int& Value);
};

