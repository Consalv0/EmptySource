#pragma once

struct FVector3;
struct FVector4;

struct FVector2 {
public:
	struct { float x, y; };

	FVector2();
	FVector2(const FVector2& Vector);
	FVector2(const FVector3& Vector);
	FVector2(const FVector4& Vector);
	FVector2(const float& Value);
	FVector2(const float& x, const float& y);

	float Magnitude() const;
	float MagnitudeSquared() const;
	FVector2 Normalized() const;

	void Normalize();
	float Cross(const FVector2& Other) const;
	float Dot(const FVector2& Other) const;

	const float* PoiterToValue() const;
	static FVector2 Lerp(const FVector2& Start, const FVector2& End, float t);

	bool operator==(const FVector2& Other);
	bool operator!=(const FVector2& Other);

	FVector2 operator+(const FVector2& Other) const;
	FVector2 operator-(const FVector2& Other) const;
	FVector2 operator-(void) const;
	FVector2 operator*(const FVector2& Other) const;
	FVector2 operator/(const FVector2& Other) const;
	FVector2 operator*(const float& Value) const;
	FVector2 operator/(const float& Value) const;

	FVector2& operator+=(const FVector2& Other);
	FVector2& operator-=(const FVector2& Other);
	FVector2& operator*=(const FVector2& Other);
	FVector2& operator/=(const FVector2& Other);
	FVector2& operator*=(const float& Value);
	FVector2& operator/=(const float& Value);
};
