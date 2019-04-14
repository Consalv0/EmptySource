#include "../include/Cubemap.h"
#include "../include/Utility/LogCore.h"

Cubemap::Cubemap(
	const int & WidthSize,
	const TextureData<UCharRGB>& Textures,
	const Graphics::ColorFormat Format,
	const Graphics::FilterMode & Filter,
	const Graphics::AddressMode & Address)
{
	Width = WidthSize;
	ColorFormat = Format;
	TextureObject = 0;

	if (!Textures.CheckDimensions(Width)) {
		Debug::Log(Debug::LogError, L"One or more texture with incorrect size in cubemap with size %d", Width);
		return;
	}

	glGenTextures(1, &TextureObject);
	SetFilterMode(Filter);
	SetAddressMode(Address);
	{
		Use();
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GetColorFormat(ColorFormat), Width, Width,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Right.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GetColorFormat(ColorFormat), Width, Width,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Left.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GetColorFormat(ColorFormat), Width, Width,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Top.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GetColorFormat(ColorFormat), Width, Width,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Bottom.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GetColorFormat(ColorFormat), Width, Width,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Front.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GetColorFormat(ColorFormat), Width, Width,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Back.PointerToValue());
	}
}

void Cubemap::Use() const {
	if (IsValid()) {
		glBindTexture(GL_TEXTURE_CUBE_MAP, TextureObject);
	}
	else {
		Debug::Log(Debug::LogWarning, L"Texture Cubemap is not valid");
	}
}

bool Cubemap::IsValid() const {
	return TextureObject != GL_FALSE && Width > 0;
}

void Cubemap::SetFilterMode(const Graphics::FilterMode & Mode) {
	FilterMode = Mode;
	Use();

	switch (Mode) {
	case Graphics::FM_MinMagLinear:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	case Graphics::FM_MinMagNearest:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		break;
	case Graphics::FM_MinLinearMagNearest:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		break;
	case Graphics::FM_MinNearestMagLinear:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	}

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void Cubemap::SetAddressMode(const Graphics::AddressMode & Mode) {
	AddressMode = Mode;
	Use();

	switch (Mode) {
	case Graphics::AM_Repeat:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT);
		break;
	case Graphics::AM_Mirror:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_MIRRORED_REPEAT);
		break;
	case Graphics::AM_Clamp:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		break;
	case Graphics::AM_Border:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
		break;
	}

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void Cubemap::Delete() {
	glDeleteTextures(1, &TextureObject);
}
