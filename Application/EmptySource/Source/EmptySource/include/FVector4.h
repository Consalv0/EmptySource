#pragma once

struct FVector2;
struct FVector3;

struct FVector4 {
public:
	struct { float x, y, z, w; };

	FVector4();
	FVector4(const FVector2& vector);
	FVector4(const FVector3& vector);
	FVector4(const FVector4& vector);
	FVector4(const float& value);
	FVector4(const float& x, const float& y, const float& z);
	FVector4(const float& x, const float& y, const float& z, const float& w);
	
	float Magnitude() const;
	float MagnitudeSquared() const;
	void Normalize();
	FVector4 Normalized() const;
	float Dot(const FVector4& other) const;

	FVector3 Vector3() const;
	FVector2 Vector2() const;
	static FVector4 Lerp(const FVector4& start, const FVector4& end, float t);

	bool operator==(const FVector4& other);
	bool operator!=(const FVector4& other);
	
	FVector4 operator+(const FVector4& other) const;
	FVector4 operator-(const FVector4& other) const;
	FVector4 operator-(void) const;
	FVector4 operator*(const float& value) const;
	FVector4 operator/(const float& value) const;
	
	FVector4& operator+=(const FVector4& other);
	FVector4& operator-=(const FVector4& other);
	FVector4& operator*=(const float& value);
	FVector4& operator/=(const float& value);
};