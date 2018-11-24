#pragma once

struct FVector2;

struct FVector3 {
public:
	float x, y, z;

	FVector3();
	FVector3(const FVector2& vector);
	FVector3(const FVector3& vector);
	FVector3(const float& value);
	FVector3(const float& x, const float& y, const float& z);
	FVector3(const float& x, const float& y);

	float Magnitude() const;
	float MagnitudeSquared() const;
	void Normalize();
	FVector3 Normalized() const;
	FVector3 Cross(const FVector3& other) const;
	float Dot(const FVector3& other) const;

	static FVector3 Lerp(const FVector3& start, const FVector3& end, float t);

	bool operator==(const FVector3& other);
	bool operator!=(const FVector3& other);

	FVector3 operator+(const FVector2& other) const;
	FVector3 operator-(const FVector2& other) const;
	FVector3 operator+(const FVector3& other) const;
	FVector3 operator-(const FVector3& other) const;
	FVector3 operator-(void) const;
	FVector3 operator*(const float& value) const;
	FVector3 operator/(const float& value) const;
	FVector3 operator*(const FVector3& other) const;
	
	FVector3& operator+=(const FVector2& other);
	FVector3& operator-=(const FVector2& other);
	FVector3& operator+=(const FVector3& other);
	FVector3& operator-=(const FVector3& other);
	FVector3& operator*=(const float& value);
	FVector3& operator/=(const float& value);
};

