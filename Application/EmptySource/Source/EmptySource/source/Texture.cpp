
#include "..\include\Texture.h"
#include "..\include\Utility\LogCore.h"
#include "..\include\Math\IntVector3.h"

Texture::Texture() {
	TextureObject = 0;
	FilterMode = Graphics::FilterMode::MinMagLinear;
	AddressMode = Graphics::AddressMode::Border;
	ColorFormat = Graphics::ColorFormat::RGB;
}

Texture3D::Texture3D(
	const IntVector3 & Size,
	const Graphics::ColorFormat Format,
	const Graphics::FilterMode & Filter,
	const Graphics::AddressMode & Address)
{
	Dimension = Size;
	ColorFormat = Format;

	glGenTextures(1, &TextureObject);
	SetFilterMode(Filter);
	SetAddressMode(Address);
	// TODO COLOR FORMATS
	{
		Use();
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, Dimension.x, Dimension.y, Dimension.z, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_3D, 0);
	}
}

IntVector3 Texture3D::GetDimension() const {
	return Dimension;
}

void Texture3D::Use() const {
	if (IsValid()) {
		glBindTexture(GL_TEXTURE_3D, TextureObject);
	} else {
		Debug::Log(Debug::LogWarning, L"Texture 3D is not valid");
	}
}

bool Texture3D::IsValid() const {
	return TextureObject != GL_FALSE && GetDimension().MagnitudeSquared() > 0;
}

void Texture3D::SetFilterMode(const Graphics::FilterMode & Mode) {
	FilterMode = Mode;
	Use();

	switch (Mode) {
	case Graphics::FilterMode::MinMagLinear:
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	case Graphics::FilterMode::MinMagNearest:
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		break;
	case Graphics::FilterMode::MinLinearMagNearest:
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		break;
	case Graphics::FilterMode::MinNearestMagLinear:
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	}

	glBindTexture(GL_TEXTURE_3D, 0);
}

void Texture3D::SetAddressMode(const Graphics::AddressMode & Mode) {
	AddressMode = Mode;
	Use();

	switch (Mode) {
	case Graphics::AddressMode::Repeat:
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
		break;
	case Graphics::AddressMode::Mirror:
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_MIRRORED_REPEAT);
		break;
	case Graphics::AddressMode::Clamp:
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		break;
	case Graphics::AddressMode::Border:
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
		break;
	}

	glBindTexture(GL_TEXTURE_3D, 0);
}

void Texture3D::Delete() {
	glDeleteTextures(1, &TextureObject);
}
