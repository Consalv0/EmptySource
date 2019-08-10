#pragma once

#include "Rendering/Texture.h"
#include "Math/MathUtility.h"
#include "Math/IntVector2.h"
#include "Math/Box2D.h"

namespace EmptySource {

	struct Texture3D : public Texture {
	private:
		//* Texture dimesions
		IntVector3 Dimension;

	public:
		//* Constructor
		Texture3D(
			const IntVector3& Size, const EColorFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode
		);

		//* Get Dimension of the texture
		IntVector3 GetDimension() const;

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