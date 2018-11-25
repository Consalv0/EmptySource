#pragma once

struct FVector2;
struct FVector3;

struct FVector4 {
public:
	union {
		struct { float x, y, z, w; };
		struct { float r, g, b, a; };
	};

	FVector4();
	FVector4(const FVector2& Vector);
	FVector4(const FVector3& Vector);
	FVector4(const FVector4& Vector);
	FVector4(const float& Value);
	FVector4(const float& x, const float& y, const float& z);
	FVector4(const float& x, const float& y, const float& z, const float& w);
	
	float Magnitude() const;
	float MagnitudeSquared() const;
	void Normalize();
	FVector4 Normalized() const;
	float Dot(const FVector4& Other) const;
	static FVector4 Lerp(const FVector4& Start, const FVector4& End, float t); 

	FVector3 Vector3() const;
	FVector2 Vector2() const;
	float & operator[](unsigned int i);
	float const& operator[](unsigned int i) const;
	const float* PoiterToValue() const;

	bool operator==(const FVector4& Other);
	bool operator!=(const FVector4& Other);
	
	FVector4 operator+(const FVector4& Other) const;
	FVector4 operator-(const FVector4& Other) const;
	FVector4 operator-(void) const;
	FVector4 operator*(const float& Value) const;
	FVector4 operator/(const float& Value) const;
	FVector4 operator*(const FVector4& Other) const;
	FVector4 operator/(const FVector4& Other) const;
	
	FVector4& operator+=(const FVector4& Other);
	FVector4& operator-=(const FVector4& Other);
	FVector4& operator*=(const FVector4& Other);
	FVector4& operator/=(const FVector4& Other);
	FVector4& operator*=(const float& Value);
	FVector4& operator/=(const float& Value);
};