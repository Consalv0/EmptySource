
#include "../include/Texture2D.h"
#include "../include/Utility/LogCore.h"

Texture2D::Texture2D(
	const IntVector2 & Size,
	const Graphics::ColorFormat Format,
	const Graphics::FilterMode & Filter,
	const Graphics::AddressMode & Address)
{
	Dimension = Size;
	ColorFormat = Format;

	glGenTextures(1, &TextureObject);
	SetFilterMode(Filter);
	SetAddressMode(Address);

	{
		Use();
		glTexImage2D(GL_TEXTURE_2D, 0, GetColorFormat(ColorFormat), Dimension.x, Dimension.y, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

Texture2D::Texture2D(
	const IntVector2 & Size,
	const Graphics::ColorFormat Format,
	const Graphics::FilterMode & Filter,
	const Graphics::AddressMode & Address,
	const Graphics::ColorFormat InputFormat,
	const unsigned int InputMode,
	const void * BufferData) 
{
	Dimension = Size;
	ColorFormat = Format;

	glGenTextures(1, &TextureObject);
	SetFilterMode(Filter);
	SetAddressMode(Address);

	{
		Use();
		glTexImage2D(GL_TEXTURE_2D, 0, GetColorFormat(ColorFormat), Dimension.x, Dimension.y, 0, GetColorFormat(InputFormat), InputMode, BufferData);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

IntVector2 Texture2D::GetDimension() const {
	return Dimension;
}

int Texture2D::GetWidth() const {
	return Dimension.x;
}

int Texture2D::GetHeight() const {
	return Dimension.y;
}

void Texture2D::Use() const {
	if (IsValid()) {
		glBindTexture(GL_TEXTURE_2D, TextureObject);
	}
	else {
		Debug::Log(Debug::LogWarning, L"Texture 2D is not valid");
	}
}

bool Texture2D::IsValid() const {
	return TextureObject != GL_FALSE && GetDimension().MagnitudeSquared() > 0;
}

void Texture2D::SetFilterMode(const Graphics::FilterMode & Mode) {
	FilterMode = Mode;
	Use();

	switch (Mode) {
	case Graphics::FM_MinMagLinear:
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	case Graphics::FM_MinMagNearest:
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		break;
	case Graphics::FM_MinLinearMagNearest:
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		break;
	case Graphics::FM_MinNearestMagLinear:
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	}

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture2D::SetAddressMode(const Graphics::AddressMode & Mode) {
	AddressMode = Mode;
	Use();

	switch (Mode) {
	case Graphics::AM_Repeat:
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		break;
	case Graphics::AM_Mirror:
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		break;
	case Graphics::AM_Clamp:
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		break;
	case Graphics::AM_Border:
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		break;
	}

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture2D::Delete() {
	glDeleteTextures(1, &TextureObject);
}
