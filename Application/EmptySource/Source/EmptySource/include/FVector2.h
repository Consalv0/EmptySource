#pragma once

struct FVector2 {
public:
	float x, y;

	FVector2();
	FVector2(const FVector2& Vector);
	FVector2(const float& Value);
	FVector2(const float& x, const float& y);

	float Magnitude() const;
	float MagnitudeSquared() const;
	FVector2 Normalized() const;

	void Normalize();
	float Cross(const FVector2& Other) const;
	float Dot(const FVector2& Other) const;

	bool operator==(const FVector2& Other);
	bool operator!=(const FVector2& Other);

	FVector2 operator+(const FVector2& Other) const;
	FVector2 operator-(const FVector2& Other) const;
	FVector2 operator-(void) const;
	FVector2 operator*(const float& Value) const;
	FVector2 operator/(const float& Value) const;
	
	FVector2& operator+=(const FVector2& Other);
	FVector2& operator-=(const FVector2& Other);
	FVector2& operator*=(const float& Value);
	FVector2& operator/=(const float& Value);
};
