
#include "..\include\Texture.h"

Texture::Texture() {
	TextureObject = 0;
	FilterMode = Graphics::FilterMode::MinMagLinear;
	AddressMode = Graphics::AddressMode::Border;
	ColorFormat = Graphics::ColorFormat::RGB;
}
