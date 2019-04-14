#pragma once

#include "../include/Texture.h"
#include "../include/Bitmap.h"

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

		inline bool CheckDimensions(const int Width) const;
	};

	Cubemap(
		const int & Width,
		const TextureData<UCharRGB>& Textures,
		const Graphics::ColorFormat Format,
		const Graphics::FilterMode & Filter,
		const Graphics::AddressMode & Address
	);

	//* Use the cubemap
	void Use() const;

	//* Check if cubemap is valid
	bool IsValid() const;

	void SetFilterMode(const Graphics::FilterMode& Mode);

	void SetAddressMode(const Graphics::AddressMode& Mode);

	//* 
	void Delete();

private:
	int Width;
};

template struct Cubemap::TextureData<UCharRGB>;

inline bool Cubemap::TextureData<UCharRGB>::CheckDimensions(const int Width) const {
	if ( Right.GetHeight() != Width ||  Right.GetWidth() != Width) return false;
	if (  Left.GetHeight() != Width ||   Left.GetWidth() != Width) return false;
	if (   Top.GetHeight() != Width ||    Top.GetWidth() != Width) return false;
	if (Bottom.GetHeight() != Width || Bottom.GetWidth() != Width) return false;
	if (  Back.GetHeight() != Width ||   Back.GetWidth() != Width) return false;
	return true;
}
