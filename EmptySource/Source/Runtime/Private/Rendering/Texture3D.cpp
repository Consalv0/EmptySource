
#include "CoreMinimal.h"
#include "Rendering/Texture3D.h"
#include "Rendering/GLFunctions.h"

namespace EmptySource {

	Texture3D::Texture3D(
		const IntVector3 & Size,
		const EColorFormat Format,
		const EFilterMode & Filter,
		const ESamplerAddressMode & Address)
	{
		Dimension = Size;
		ColorFormat = Format;

		glGenTextures(1, &TextureObject);
		Use();
		SetFilterMode(Filter);
		SetSamplerAddressMode(Address);

		{
			glTexImage3D(
				GL_TEXTURE_3D, 0, (int)ColorFormat, Dimension.x, Dimension.y, Dimension.z, 0,
				GL_RGBA, GL_FLOAT, NULL
			);
			glBindTexture(GL_TEXTURE_3D, 0);
		}
	}

	IntVector3 Texture3D::GetDimension() const {
		return Dimension;
	}

	void Texture3D::GenerateMipMaps() {
		if (IsValid()) {
			Use();
			bLods = true;
			SetFilterMode(FilterMode);
			glGenerateMipmap(GL_TEXTURE_3D);
		}
	}

	void Texture3D::Use() const {
		if (IsValid()) {
			glBindTexture(GL_TEXTURE_3D, TextureObject);
		}
		else {
			LOG_CORE_WARN(L"Texture 3D is not valid");
		}
	}

	void Texture3D::Deuse() const {
		glBindTexture(GL_TEXTURE_3D, 0);
	}

	bool Texture3D::IsValid() const {
		return TextureObject != GL_FALSE && GetDimension().MagnitudeSquared() > 0;
	}

	void Texture3D::SetFilterMode(const EFilterMode & Mode) {
		FilterMode = Mode;

		switch (Mode) {
		case FM_MinMagLinear:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, bLods ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
			break;
		case FM_MinMagNearest:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, bLods ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
			break;
		case FM_MinLinearMagNearest:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, bLods ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR);
			break;
		case FM_MinNearestMagLinear:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, bLods ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST);
			break;
		}
	}

	void Texture3D::SetSamplerAddressMode(const ESamplerAddressMode & Mode) {
		AddressMode = Mode;

		switch (Mode) {
		case SAM_Repeat:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
			break;
		case SAM_Mirror:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_MIRRORED_REPEAT);
			break;
		case SAM_Clamp:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			break;
		case SAM_Border:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
			break;
		}
	}

	void Texture3D::Delete() {
		glDeleteTextures(1, &TextureObject);
	}

}