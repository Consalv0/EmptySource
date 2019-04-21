#pragma once

struct Vector4;
struct Vector3;
struct Quaternion;

struct Matrix4x4 {

public:
	union {
		struct { Vector4 m0, m1, m2, m3; };
		struct { Vector4 x, y, z, w; };
	};

	HOST_DEVICE FORCEINLINE Matrix4x4();
	HOST_DEVICE FORCEINLINE Matrix4x4(const Matrix4x4& Other);
	HOST_DEVICE FORCEINLINE Matrix4x4(const Vector4& Row0, const Vector4& Row1, const Vector4& Row2, const Vector4 Row3); 
	HOST_DEVICE FORCEINLINE Matrix4x4(
		float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33
	);
	
	HOST_DEVICE inline static Matrix4x4 Identity();
	
	//* Creates a perspective matrix, FOV is the aperture angle in radians
	HOST_DEVICE inline static Matrix4x4 Perspective(const float& Aperture, const float& Aspect, const float& Near, const float& Far);
	//* Creates a orthographic matrix
	HOST_DEVICE inline static Matrix4x4 Orthographic(const float& Left, const float& Right, const float& Bottom, const float& Top);
	//* Creates a orthographic matrix with zClamping
	HOST_DEVICE inline static Matrix4x4 Orthographic(const float& Left, const float& Right, const float& Bottom, const float& Top, const float& Near, const float& Far);
	HOST_DEVICE inline static Matrix4x4 LookAt(const Vector3& Position, const Vector3& Direction, const Vector3& Up);
	HOST_DEVICE inline static Matrix4x4 Translation(const Vector3& Vector);
	HOST_DEVICE inline static Matrix4x4 Scaling(const Vector3& Vector);
	HOST_DEVICE inline static Matrix4x4 Rotation(const Vector3& Axis, const float& Angle);
	HOST_DEVICE inline static Matrix4x4 Rotation(const Vector3& EulerAngles);
	HOST_DEVICE inline static Matrix4x4 Rotation(const Quaternion& Quat);
	
	HOST_DEVICE inline void Transpose();
	HOST_DEVICE inline Matrix4x4 Transposed() const;
	HOST_DEVICE inline Matrix4x4 Inversed() const;
	
	HOST_DEVICE inline Vector4 Row(const int& i) const;
	HOST_DEVICE inline Vector4 Column(const int& i) const;
	
	HOST_DEVICE inline Vector4 & operator[](unsigned int i);
	HOST_DEVICE inline Vector4 const& operator[](unsigned int i) const;
	HOST_DEVICE inline const float* PointerToValue(void) const;

	HOST_DEVICE inline Vector3 MultiplyPoint(const Vector3& Vector) const;
	HOST_DEVICE inline Vector3 MultiplyVector(const Vector3& Vector) const;

	HOST_DEVICE FORCEINLINE Matrix4x4 operator*(const Matrix4x4& Other) const;
	HOST_DEVICE FORCEINLINE Vector4 operator*(const Vector4& Vector) const;
	HOST_DEVICE FORCEINLINE Vector3 operator*(const Vector3& Vector) const;
	HOST_DEVICE FORCEINLINE Matrix4x4 operator*(const float& Value) const;
	HOST_DEVICE FORCEINLINE Matrix4x4 operator/(const float& Value) const;
};

#include "../Math/Matrix4x4.inl"
