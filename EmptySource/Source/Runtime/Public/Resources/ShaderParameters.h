#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/Texture.h"

namespace EmptySource {

	enum ShaderPropertyFlags {
		SPFlags_None = 0x0,
		SPFlags_IsInternal = 1 << 0,
		SPFlags_IsColor = 1 << 1
	};

	struct ShaderProperty {
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
				TexturePtr Texture;
				TArray<Matrix4x4> Matrix4x4Array;
				TArray<float> FloatArray;
				TArray<Vector2> Float2DArray;
				TArray<Vector3> Float3DArray;
				TArray<Vector4> Float4DArray;
			};

			PropertyValue() { memset(this, 0, sizeof(PropertyValue)); }
			PropertyValue(const PropertyValue & Other)              : PropertyValue() { *this = Other; }
			PropertyValue(const TArray<Matrix4x4> & Matrix4x4Array) : Type(EShaderPropertyType::Matrix4x4Array), Matrix4x4Array(Matrix4x4Array) { }
			PropertyValue(const TArray<float> & FloatArray)         : Type(EShaderPropertyType::FloatArray), FloatArray(FloatArray) { }
			PropertyValue(const TArray<Vector2> & Float2DArray)     : Type(EShaderPropertyType::Float2DArray), Float2DArray(Float2DArray) { }
			PropertyValue(const TArray<Vector3> & Float3DArray)     : Type(EShaderPropertyType::Float3DArray), Float3DArray(Float3DArray) { }
			PropertyValue(const TArray<Vector4> & Float4DArray)     : Type(EShaderPropertyType::Float4DArray), Float4DArray(Float4DArray) { }
			PropertyValue(const TArray<int> & IntArray)             : Type(EShaderPropertyType::IntArray), IntArray(IntArray) { }
			PropertyValue(const Matrix4x4 & Matrix)                 : Type(EShaderPropertyType::Matrix4x4), Mat4x4(Matrix) { }
			PropertyValue(const float & Float)                      : Type(EShaderPropertyType::Float), Float(Float) { }
			PropertyValue(const Vector2 & Float2D)                  : Type(EShaderPropertyType::Float2D), Float2D(Float2D) { }
			PropertyValue(const Vector3 & Float3D)                  : Type(EShaderPropertyType::Float3D), Float3D(Float3D) { }
			PropertyValue(const Vector4 & Float4D)                  : Type(EShaderPropertyType::Float4D), Float4D(Float4D) { }
			PropertyValue(const int & Int)                          : Type(EShaderPropertyType::Int), Int(Int) { }
			PropertyValue(ETextureDimension Type,
				const TexturePtr& Text)                             : Type(Type == ETextureDimension::Cubemap ? EShaderPropertyType::Cubemap :
				                                                           EShaderPropertyType::Texture2D), Texture(Text) { }
			~PropertyValue() {}

			void * PointerToValue() const {
				switch (Type) {
				case EShaderPropertyType::Matrix4x4:      return (void *)Mat4x4.PointerToValue();
				case EShaderPropertyType::Float:          return (void *)&Float;
				case EShaderPropertyType::Float2D:        return (void *)Float2D.PointerToValue();
				case EShaderPropertyType::Float3D:        return (void *)Float3D.PointerToValue();
				case EShaderPropertyType::Float4D:        return (void *)Float4D.PointerToValue();
				case EShaderPropertyType::Int:            return (void *)&Int;
				case EShaderPropertyType::Matrix4x4Array: return Matrix4x4Array.empty() ? NULL : (void *)Matrix4x4Array[0].PointerToValue();
				case EShaderPropertyType::FloatArray:     return FloatArray.empty()     ? NULL : (void *)&FloatArray[0];
				case EShaderPropertyType::Float2DArray:   return Float2DArray.empty()   ? NULL : (void *)Float2DArray[0].PointerToValue();
				case EShaderPropertyType::Float3DArray:   return Float3DArray.empty()   ? NULL : (void *)Float3DArray[0].PointerToValue();
				case EShaderPropertyType::Float4DArray:   return Float4DArray.empty()   ? NULL : (void *)Float4DArray[0].PointerToValue();
				case EShaderPropertyType::IntArray:       return IntArray.empty()       ? NULL : (void *)&IntArray[0];
				case EShaderPropertyType::Texture2D:
				case EShaderPropertyType::Cubemap:
				case EShaderPropertyType::None:
				default:
					return 0;
					break;
				}
			}

			PropertyValue& operator=(const PropertyValue & Other) {
				Type = Other.Type;
				switch (Other.Type) {
				case EShaderPropertyType::Matrix4x4:      Mat4x4 = Other.Mat4x4; break;
				case EShaderPropertyType::Float:          Float = Other.Float; break;
				case EShaderPropertyType::Float2D:        Float2D = Other.Float2D; break;
				case EShaderPropertyType::Float3D:        Float3D = Other.Float3D; break;
				case EShaderPropertyType::Float4D:        Float4D = Other.Float4D; break;
				case EShaderPropertyType::Int:            Int = Other.Int; break;
				case EShaderPropertyType::Matrix4x4Array: Matrix4x4Array = Other.Matrix4x4Array; break;
				case EShaderPropertyType::FloatArray:     FloatArray = Other.FloatArray; break;
				case EShaderPropertyType::Float2DArray:   Float2DArray = Other.Float2DArray; break;
				case EShaderPropertyType::Float3DArray:   Float3DArray = Other.Float3DArray; break;
				case EShaderPropertyType::Float4DArray:   Float4DArray = Other.Float4DArray; break;
				case EShaderPropertyType::IntArray:       IntArray = Other.IntArray; break;
				case EShaderPropertyType::Texture2D:
				case EShaderPropertyType::Cubemap:        Texture = Other.Texture; break;
				case EShaderPropertyType::None:
				default:
					break;
				}

				return *this;
			}

			inline const EShaderPropertyType & GetType() const { return Type; };

		private:
			EShaderPropertyType Type;
		} Value;

		ShaderProperty(const NString & Name, const ShaderProperty::PropertyValue & Value, int SPFalgs = 0)
			: Name(Name), Value(Value), Flags(SPFalgs) {
		};

		ShaderProperty(const ShaderProperty& Other) {
			Value = Other.Value;
			Name = Other.Name;
			Flags = SPFlags_None;
		};

		inline bool IsColor() const { return Flags & SPFlags_IsColor; }

		inline bool IsInternal() const { return Flags & SPFlags_IsInternal; }

		ShaderProperty& operator=(const ShaderProperty & Other) {
			Value = Other.Value;
			Name = Other.Name;
			Flags = Other.Flags;
			return *this;
		}

		ShaderProperty& operator=(const ShaderProperty::PropertyValue & Other) {
			Value = Other;
			return *this;
		}
	};

}