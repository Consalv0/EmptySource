#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Resources/TextureResource.h"

namespace EmptySource {

	enum ShaderPropertyFlags {
		SPFlags_None = 0x0,
		SPFlags_IsInternal = 1 << 0,
		SPFlags_IsColor = 1 << 1
	};

	struct ShaderParameters {
		NString Name;
		unsigned int Flags;

		struct PropertyValue {
			union {
				TArray<int> IntArray;
				Matrix4x4 Mat4x4;
				float Float;
				Vector2 Float2D;
				Vector3 Float3D;
				Vector4 Float4D;
				int Int;
				RTexturePtr Texture;
				TArray<Matrix4x4> Matrix4x4Array;
				TArray<float> FloatArray;
				TArray<Vector2> Float2DArray;
				TArray<Vector3> Float3DArray;
				TArray<Vector4> Float4DArray;
			};

			PropertyValue() { memset(this, 0, sizeof(PropertyValue)); }
			PropertyValue(const PropertyValue & Other)              : PropertyValue() { *this = Other; }
			PropertyValue(const TArray<Matrix4x4> & Matrix4x4Array) : Type(EShaderUniformType::Matrix4x4Array), Matrix4x4Array(Matrix4x4Array) { }
			PropertyValue(const TArray<float> & FloatArray)         : Type(EShaderUniformType::FloatArray), FloatArray(FloatArray) { }
			PropertyValue(const TArray<Vector2> & Float2DArray)     : Type(EShaderUniformType::Float2DArray), Float2DArray(Float2DArray) { }
			PropertyValue(const TArray<Vector3> & Float3DArray)     : Type(EShaderUniformType::Float3DArray), Float3DArray(Float3DArray) { }
			PropertyValue(const TArray<Vector4> & Float4DArray)     : Type(EShaderUniformType::Float4DArray), Float4DArray(Float4DArray) { }
			PropertyValue(const TArray<int> & IntArray)             : Type(EShaderUniformType::IntArray), IntArray(IntArray) { }
			PropertyValue(const Matrix4x4 & Matrix)                 : Type(EShaderUniformType::Matrix4x4), Mat4x4(Matrix) { }
			PropertyValue(const float & Float)                      : Type(EShaderUniformType::Float), Float(Float) { }
			PropertyValue(const Vector2 & Float2D)                  : Type(EShaderUniformType::Float2D), Float2D(Float2D) { }
			PropertyValue(const Vector3 & Float3D)                  : Type(EShaderUniformType::Float3D), Float3D(Float3D) { }
			PropertyValue(const Vector4 & Float4D)                  : Type(EShaderUniformType::Float4D), Float4D(Float4D) { }
			PropertyValue(const int & Int)                          : Type(EShaderUniformType::Int), Int(Int) { }
			PropertyValue(ETextureDimension Type,
				const RTexturePtr& Text)                            : Type(Type == ETextureDimension::Cubemap ? EShaderUniformType::Cubemap :
				                                                           EShaderUniformType::Texture2D), Texture(Text) { }
			~PropertyValue() {}

			void * PointerToValue() const {
				switch (Type) {
				case EShaderUniformType::Matrix4x4:      return (void *)Mat4x4.PointerToValue();
				case EShaderUniformType::Float:          return (void *)&Float;
				case EShaderUniformType::Float2D:        return (void *)Float2D.PointerToValue();
				case EShaderUniformType::Float3D:        return (void *)Float3D.PointerToValue();
				case EShaderUniformType::Float4D:        return (void *)Float4D.PointerToValue();
				case EShaderUniformType::Int:            return (void *)&Int;
				case EShaderUniformType::Matrix4x4Array: return Matrix4x4Array.empty() ? NULL : (void *)Matrix4x4Array[0].PointerToValue();
				case EShaderUniformType::FloatArray:     return FloatArray.empty()     ? NULL : (void *)&FloatArray[0];
				case EShaderUniformType::Float2DArray:   return Float2DArray.empty()   ? NULL : (void *)Float2DArray[0].PointerToValue();
				case EShaderUniformType::Float3DArray:   return Float3DArray.empty()   ? NULL : (void *)Float3DArray[0].PointerToValue();
				case EShaderUniformType::Float4DArray:   return Float4DArray.empty()   ? NULL : (void *)Float4DArray[0].PointerToValue();
				case EShaderUniformType::IntArray:       return IntArray.empty()       ? NULL : (void *)&IntArray[0];
				case EShaderUniformType::Texture2D:
				case EShaderUniformType::Cubemap:
				case EShaderUniformType::None:
				default:
					return 0;
					break;
				}
			}

			PropertyValue& operator=(const PropertyValue & Other) {
				Type = Other.Type;
				switch (Other.Type) {
				case EShaderUniformType::Matrix4x4:      Mat4x4 = Other.Mat4x4; break;
				case EShaderUniformType::Float:          Float = Other.Float; break;
				case EShaderUniformType::Float2D:        Float2D = Other.Float2D; break;
				case EShaderUniformType::Float3D:        Float3D = Other.Float3D; break;
				case EShaderUniformType::Float4D:        Float4D = Other.Float4D; break;
				case EShaderUniformType::Int:            Int = Other.Int; break;
				case EShaderUniformType::Matrix4x4Array: Matrix4x4Array = Other.Matrix4x4Array; break;
				case EShaderUniformType::FloatArray:     FloatArray = Other.FloatArray; break;
				case EShaderUniformType::Float2DArray:   Float2DArray = Other.Float2DArray; break;
				case EShaderUniformType::Float3DArray:   Float3DArray = Other.Float3DArray; break;
				case EShaderUniformType::Float4DArray:   Float4DArray = Other.Float4DArray; break;
				case EShaderUniformType::IntArray:       IntArray = Other.IntArray; break;
				case EShaderUniformType::Texture2D:
				case EShaderUniformType::Cubemap:        Texture = Other.Texture; break;
				case EShaderUniformType::None:
				default:
					break;
				}

				return *this;
			}

			inline const EShaderUniformType & GetType() const { return Type; };

		private:
			EShaderUniformType Type;
		} Value;

		ShaderParameters(const NString & Name, const ShaderParameters::PropertyValue & Value, int SPFalgs = 0)
			: Name(Name), Value(Value), Flags(SPFalgs) {
		};

		ShaderParameters(const ShaderParameters& Other) {
			Value = Other.Value;
			Name = Other.Name;
			Flags = Other.Flags;
		};

		inline bool IsColor() const { return Flags & SPFlags_IsColor; }

		inline bool IsInternal() const { return Flags & SPFlags_IsInternal; }

		ShaderParameters& operator=(const ShaderParameters & Other) {
			Value = Other.Value;
			Name = Other.Name;
			Flags = Other.Flags;
			return *this;
		}

		ShaderParameters& operator=(const ShaderParameters::PropertyValue & Other) {
			Value = Other;
			return *this;
		}
	};

}