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
		bool bColor;
		bool bInternal;
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
			PropertyValue(const PropertyValue & Other) : PropertyValue() { *this = Other; }
			PropertyValue(const TArray<Matrix4x4> & Matrix4x4Array) : Type(EShaderPropertyType::Matrix4x4Array), Matrix4x4Array(Matrix4x4Array) { }
			PropertyValue(const TArray<float> & FloatArray) : Type(EShaderPropertyType::FloatArray), FloatArray(FloatArray) { }
			PropertyValue(const TArray<Vector2> & Float2DArray) : Type(EShaderPropertyType::Float2DArray), Float2DArray(Float2DArray) { }
			PropertyValue(const TArray<Vector3> & Float3DArray) : Type(EShaderPropertyType::Float3DArray), Float3DArray(Float3DArray) { }
			PropertyValue(const TArray<Vector4> & Float4DArray) : Type(EShaderPropertyType::Float4DArray), Float4DArray(Float4DArray) { }
			PropertyValue(const TArray<int> & IntArray) : Type(EShaderPropertyType::IntArray), IntArray(IntArray) { }
			PropertyValue(const Matrix4x4 & Matrix) : Type(EShaderPropertyType::Matrix4x4), Mat4x4(Matrix) { }
			PropertyValue(const float & Float) : Type(EShaderPropertyType::Float), Float(Float) { }
			PropertyValue(const Vector2 & Float2D) : Type(EShaderPropertyType::Float2D), Float2D(Float2D) { }
			PropertyValue(const Vector3 & Float3D) : Type(EShaderPropertyType::Float3D), Float3D(Float3D) { }
			PropertyValue(const Vector4 & Float4D) : Type(EShaderPropertyType::Float4D), Float4D(Float4D) { }
			PropertyValue(const int & Int) : Type(EShaderPropertyType::Int), Int(Int) { }
			PropertyValue(ETextureDimension Type, const TexturePtr Text)
				: Type(Type == ETextureDimension::Cubemap ? EShaderPropertyType::Cubemap : EShaderPropertyType::Texture2D), Texture(Text) { }
			~PropertyValue() {}

			void * PointerToValue() const {
				switch (Type) {
				case EmptySource::EShaderPropertyType::Matrix4x4:
					return (void *)Mat4x4.PointerToValue();
				case EmptySource::EShaderPropertyType::Float:
					return (void *)&Float;
				case EmptySource::EShaderPropertyType::Float2D:
					return (void *)Float2D.PointerToValue();
				case EmptySource::EShaderPropertyType::Float3D:
					return (void *)Float3D.PointerToValue();
				case EmptySource::EShaderPropertyType::Float4D:
					return (void *)Float4D.PointerToValue();
				case EmptySource::EShaderPropertyType::Int:
					return (void *)&Int;
				case EmptySource::EShaderPropertyType::Matrix4x4Array:
					return Matrix4x4Array.empty() ? 0 : (void *)Matrix4x4Array[0].PointerToValue();
				case EmptySource::EShaderPropertyType::FloatArray:
					return FloatArray.empty() ? 0 : (void *)&FloatArray[0];
				case EmptySource::EShaderPropertyType::Float2DArray:
					return Float2DArray.empty() ? 0 : (void *)Float2DArray[0].PointerToValue();
				case EmptySource::EShaderPropertyType::Float3DArray:
					return Float3DArray.empty() ? 0 : (void *)Float3DArray[0].PointerToValue();
				case EmptySource::EShaderPropertyType::Float4DArray:
					return Float4DArray.empty() ? 0 : (void *)Float4DArray[0].PointerToValue();
				case EmptySource::EShaderPropertyType::IntArray:
					return IntArray.empty() ? 0 : (void *)&IntArray[0];
				case EmptySource::EShaderPropertyType::Texture2D:
				case EmptySource::EShaderPropertyType::Cubemap:
				case EmptySource::EShaderPropertyType::None:
				default:
					return 0;
					break;
				}
			}

			PropertyValue& operator=(const PropertyValue & Other) {
				Type = Other.Type;
				switch (Other.Type) {
				case EmptySource::EShaderPropertyType::Matrix4x4:
					Mat4x4 = Other.Mat4x4; break;
				case EmptySource::EShaderPropertyType::Float:
					Float = Other.Float; break;
				case EmptySource::EShaderPropertyType::Float2D:
					Float2D = Other.Float2D; break;
				case EmptySource::EShaderPropertyType::Float3D:
					Float3D = Other.Float3D; break;
				case EmptySource::EShaderPropertyType::Float4D:
					Float4D = Other.Float4D; break;
				case EmptySource::EShaderPropertyType::Int:
					Int = Other.Int; break;
				case EmptySource::EShaderPropertyType::Matrix4x4Array:
					Matrix4x4Array = Other.Matrix4x4Array; break;
				case EmptySource::EShaderPropertyType::FloatArray:
					FloatArray = Other.FloatArray; break;
				case EmptySource::EShaderPropertyType::Float2DArray:
					Float2DArray = Other.Float2DArray; break;
				case EmptySource::EShaderPropertyType::Float3DArray:
					Float3DArray = Other.Float3DArray; break;
				case EmptySource::EShaderPropertyType::Float4DArray:
					Float4DArray = Other.Float4DArray; break;
				case EmptySource::EShaderPropertyType::IntArray:
					IntArray = Other.IntArray; break;
				case EmptySource::EShaderPropertyType::Texture2D:
				case EmptySource::EShaderPropertyType::Cubemap:
					Texture = Other.Texture; break;
				case EmptySource::EShaderPropertyType::None:
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
			: Name(Name), Value(Value), bColor(SPFalgs & SPFlags_IsColor), bInternal(SPFalgs & SPFlags_IsInternal) {
		};

		ShaderProperty(const ShaderProperty& Other) {
			Value = Other.Value;
			Name = Other.Name;
			bColor = Other.bColor;
			bInternal = Other.bInternal;
		};

		ShaderProperty& operator=(const ShaderProperty & Other) {
			Value = Other.Value;
			Name = Other.Name;
			bColor = Other.bColor;
			bInternal = Other.bInternal;
			return *this;
		}

		ShaderProperty& operator=(const ShaderProperty::PropertyValue & Other) {
			Value = Other;
			return *this;
		}
	};

}