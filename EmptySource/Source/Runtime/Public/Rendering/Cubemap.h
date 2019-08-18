#pragma once

#include "Rendering/Texture.h"
#include "Rendering/Bitmap.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	struct Cubemap : Texture {
	public:
		template<typename T>
		struct TextureData {
			// GL_TEXTURE_CUBE_MAP_POSITIVE_X Right
			Bitmap<T> Right;
			// GL_TEXTURE_CUBE_MAP_NEGATIVE_X Left
			Bitmap<T> Left;
			// GL_TEXTURE_CUBE_MAP_POSITIVE_Y Top
			Bitmap<T> Top;
			// GL_TEXTURE_CUBE_MAP_NEGATIVE_Y Bottom
			Bitmap<T> Bottom;
			// GL_TEXTURE_CUBE_MAP_POSITIVE_Z Back
			Bitmap<T> Back;
			// GL_TEXTURE_CUBE_MAP_NEGATIVE_Z Front
			Bitmap<T> Front;

			inline bool CheckDimensions(const int Width) const {
				if (Right.GetHeight() != Width || Right.GetWidth() != Width) return false;
				if (Left.GetHeight() != Width || Left.GetWidth() != Width) return false;
				if (Top.GetHeight() != Width || Top.GetWidth() != Width) return false;
				if (Bottom.GetHeight() != Width || Bottom.GetWidth() != Width) return false;
				if (Back.GetHeight() != Width || Back.GetWidth() != Width) return false;
				return true;
			}
		};

		Cubemap(
			const int & Width,
			const EColorFormat & Format,
			const EFilterMode & Filter,
			const ESamplerAddressMode & AddressMode
		);

		//* Use the cubemap
		void Use() const;

		//* Deuse the texture
		void Deuse() const;

		//* Check if cubemap is valid
		bool IsValid() const;

		int GetWidth() const;

		float GetMipmapCount() const;
		void GenerateMipMaps();

		bool CalculateIrradianceMap() const;

		static bool FromCube(Cubemap& Map, const TextureData<UCharRGB>& Textures);
		static bool FromHDRCube(Cubemap& Map, const TextureData<FloatRGB>& Textures);

		static bool FromEquirectangular(Cubemap& Map, struct Texture2D* Equirectangular, ShaderPtr ShaderConverter);
		static bool FromHDREquirectangular(Cubemap& Map, struct Texture2D* Equirectangular, ShaderPtr ShaderConverter);

		//* 
		void Delete();

	private:
		void SetFilterMode(const EFilterMode& FilterMode);

		void SetSamplerAddressMode(const ESamplerAddressMode& SamplerAddressMode);

		int Width;
	};

	template struct Cubemap::TextureData<UCharRGB>;
	template struct Cubemap::TextureData<FloatRGB>;

}