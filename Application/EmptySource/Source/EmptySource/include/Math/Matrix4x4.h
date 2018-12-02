#pragma once

struct Vector4;
struct Vector3;

struct Matrix4x4 {

public:
	struct {
		Vector4 m0, m1, m2, m3;
	};

	FORCEINLINE Matrix4x4();
	FORCEINLINE Matrix4x4(const Matrix4x4& Other);
	FORCEINLINE Matrix4x4(const Vector4& Row0, const Vector4& Row1, const Vector4& Row2, const Vector4 Row3);

	inline static Matrix4x4 Identity();

	//* Creates a perspective matrix, FOV is the aperture angle in radians
	inline static Matrix4x4 Perspective(const float& Aperture, const float& Aspect, const float& Near, const float& Far);
	inline static Matrix4x4 LookAt(const Vector3& Position, const Vector3& Direction, const Vector3& Up);
	inline static Matrix4x4 Translate(const Vector3& Vector);

	inline void Transpose();
	inline Matrix4x4 Transposed() const;
	inline Matrix4x4 Inversed() const;

	inline Vector4 Row(const int& i) const;
	inline Vector4 Column(const int& i) const;

	inline Vector4 & operator[](unsigned int i);
	inline Vector4 const& operator[](unsigned int i) const;
	inline const float* PointerToValue(void) const;
	
	FORCEINLINE Matrix4x4 operator*(const Matrix4x4& Other) const;
	FORCEINLINE Vector4 operator*(const Vector4& Vector) const;
	FORCEINLINE Vector3 operator*(const Vector3& Vector) const;
	FORCEINLINE Matrix4x4 operator*(const float& Value) const;
	FORCEINLINE Matrix4x4 operator/(const float& Value) const;
};

#include "..\Math\Matrix4x4.inl"