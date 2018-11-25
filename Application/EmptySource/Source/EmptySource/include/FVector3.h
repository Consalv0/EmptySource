#pragma once

struct FVector2;
struct FVector4;

struct FVector3 {
public:
	struct { float x, y, z; };

	FVector3();
	FVector3(const FVector2& Vector);
	FVector3(const FVector3& Vector);
	FVector3(const FVector4& Vector);
	FVector3(const float& Value);
	FVector3(const float& x, const float& y, const float& z);
	FVector3(const float& x, const float& y);

	float Magnitude() const;
	float MagnitudeSquared() const;
	void Normalize();
	FVector3 Normalized() const;
	FVector3 Cross(const FVector3& Other) const;
	float Dot(const FVector3& Other) const;

	FVector2 Vector2() const;
	const float* PoiterToValue() const;
	static FVector3 Lerp(const FVector3& start, const FVector3& end, float t);

	bool operator==(const FVector3& Other);
	bool operator!=(const FVector3& Other);
	
	FVector3 operator+(const FVector3& Other) const;
	FVector3 operator-(const FVector3& Other) const;
	FVector3 operator-(void) const;
	FVector3 operator*(const FVector3& Other) const;
	FVector3 operator/(const FVector3& Other) const;
	FVector3 operator*(const float& Value) const;
	FVector3 operator/(const float& Value) const;

	FVector3& operator+=(const FVector3& Other);
	FVector3& operator-=(const FVector3& Other);
	FVector3& operator*=(const FVector3& Other);
	FVector3& operator/=(const FVector3& Other);
	FVector3& operator*=(const float& Value);
	FVector3& operator/=(const float& Value);
};

