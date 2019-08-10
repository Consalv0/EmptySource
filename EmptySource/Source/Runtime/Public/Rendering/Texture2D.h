#pragma once

#include "Rendering/Texture.h"
#include "Math/MathUtility.h"
#include "Math/IntVector2.h"
#include "Math/Box2D.h"

namespace EmptySource {

	struct Texture2D : public Texture {
	private:
		//* Texture dimesions
		IntVector2 Dimension;

	public:
		//* Constructor
		Texture2D(
			const IntVector2& Size, const EColorFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode
		);

		Texture2D(
			const IntVector2& Size, const EColorFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode,
			const EColorFormat InputFormat, const void* BufferData
		);

		//* Get Dimension of the texture
		IntVector2 GetDimension() const;

		int GetWidth() const;

		int GetHeight() const;

		void GenerateMipMaps();

		//* Use the texture
		void Use() const;

		//* Deuse the texture
		void Deuse() const;

		//* Check if texture is valid
		bool IsValid() const;

		void SetFilterMode(const EFilterMode& Mode);

		void SetSamplerAddressMode(const ESamplerAddressMode& Mode);

		//* 
		void Delete();
	};

}