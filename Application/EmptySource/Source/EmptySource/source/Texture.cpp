
#include "..\include\Texture.h"

Texture::Texture() {
	TextureObject = 0;
	FilterMode = Graphics::FM_MinMagLinear;
	AddressMode = Graphics::AM_Border;
	ColorFormat = Graphics::CF_RGB;
}
