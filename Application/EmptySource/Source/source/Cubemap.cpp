#include "../include/Cubemap.h"
#include "../include/Utility/LogCore.h"
#include "../include/Texture2D.h"
#include "../include/RenderTarget.h"
#include "../include/Material.h"
#include "../include/Mesh.h"
#include "../include/Math/Matrix4x4.h"
#include "../include/Math/MathUtility.h"

Cubemap::Cubemap(
	const int & WidthSize,
	const Graphics::ColorFormat & Format,
	const Graphics::FilterMode & Filter,
	const Graphics::AddressMode & Address)
{
	Width = WidthSize;
	ColorFormat = Format;
	TextureObject = 0;

	if (WidthSize <= 0) {
		Debug::Log(Debug::LogError, L"One or more texture with incorrect size in cubemap with size %d", Width);
		return;
	}

	glGenTextures(1, &TextureObject);
	Use();
	SetFilterMode(Filter);
	SetAddressMode(Address);
	Deuse();
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

int Cubemap::GetWidth() const {
	return Width;
}

float Cubemap::GetMipmapCount() const {
	return log2f((float)Width);
}

void Cubemap::GenerateMipMaps() {
	if (IsValid()) {
		Use();
		bLods = true;
		SetFilterMode(FilterMode);
		glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
	}
}

bool Cubemap::CalculateIrradianceMap() const {
	return true;
}

void Cubemap::SetFilterMode(const Graphics::FilterMode & Mode) {
	FilterMode = Mode;

	switch (Mode) {
	case Graphics::FM_MinMagLinear:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, bLods ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
		break;
	case Graphics::FM_MinMagNearest:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, bLods ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
		break;
	case Graphics::FM_MinLinearMagNearest:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, bLods ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR);
		break;
	case Graphics::FM_MinNearestMagLinear:
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, bLods ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST);
		break;
	}
}

void Cubemap::SetAddressMode(const Graphics::AddressMode & Mode) {
	AddressMode = Mode;

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
}

bool Cubemap::FromCube(Cubemap& Map, const TextureData<UCharRGB>& Textures) {
	if (!Map.IsValid()) return false;

	if (!Textures.CheckDimensions(Map.Width) || Map.Width <= 0) {
		return false;
	}

	Map.Use();
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Right.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Left.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Top.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Bottom.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Front.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Back.PointerToValue());

	Map.GenerateMipMaps();
	Map.Deuse();
	return true;
}

// TODO
bool Cubemap::FromHDRCube(Cubemap & Map, const TextureData<FloatRGB>& Textures) {
	if (!Map.IsValid()) return false;

	if (!Textures.CheckDimensions(Map.Width) || Map.Width <= 0) {
		return false;
	}

	Map.Use();
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_FLOAT, Textures.Right.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_FLOAT, Textures.Left.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_FLOAT, Textures.Top.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_FLOAT, Textures.Bottom.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_FLOAT, Textures.Front.PointerToValue());
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GetColorFormat(Map.ColorFormat), Map.Width, Map.Width,
		0, GL_RGB, GL_FLOAT, Textures.Back.PointerToValue());

	Map.GenerateMipMaps();
	Map.Deuse();
	return true;
}

bool Cubemap::FromEquirectangular(Cubemap & Map, Texture2D * Equirectangular, Mesh * CubeModel, ShaderProgram * ShaderConverter) {
	if (!Map.IsValid()) return false;

	Material EquirectangularToCubemapMaterial = Material();
	EquirectangularToCubemapMaterial.SetShaderProgram(ShaderConverter);
	EquirectangularToCubemapMaterial.CullMode = Graphics::CM_None;

	Map.Use();
	{
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		Map.GenerateMipMaps();
	}

	static const Matrix4x4 CaptureProjection = Matrix4x4::Perspective(90.F * MathConstants::DegreeToRad, 1.F, 0.1F, 10.F);
	static const Matrix4x4 CaptureViews[] = {
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(-1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  1.F,  0.F), Vector3(0.F,  0.F,  1.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F, -1.F,  0.F), Vector3(0.F,  0.F, -1.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F,  1.F), Vector3(0.F, -1.F,  0.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F, -1.F), Vector3(0.F, -1.F,  0.F))
	};

	GLuint ModelMatrixBuffer;
	glGenBuffers(1, &ModelMatrixBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
	RenderTarget Renderer = RenderTarget();
	Renderer.SetUpBuffers();

	// --- Convert HDR equirectangular environment map to cubemap equivalent
	EquirectangularToCubemapMaterial.Use();
	EquirectangularToCubemapMaterial.SetTexture2D("_EquirectangularMap", Equirectangular, 0);
	EquirectangularToCubemapMaterial.SetMatrix4x4Array("_ProjectionMatrix", CaptureProjection.PointerToValue());
	EquirectangularToCubemapMaterial.CullMode = Graphics::CM_Front;

	Renderer.Resize(Map.Width, Map.Width);
	for (unsigned int i = 0; i < 6; ++i) {
		EquirectangularToCubemapMaterial.SetMatrix4x4Array("_ViewMatrix", CaptureViews[i].PointerToValue());

		CubeModel->SetUpBuffers();
		CubeModel->BindVertexArray();
		EquirectangularToCubemapMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
		Renderer.PrepareTexture(&Map, i);
		Renderer.Clear();

		CubeModel->DrawElement();
	}
	Map.GenerateMipMaps();
	Map.Deuse();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	Renderer.Delete();
	glDeleteBuffers(1, &ModelMatrixBuffer);
	return true;
}

bool Cubemap::FromHDREquirectangular(Cubemap & Map, Texture2D * Equirectangular, Mesh * CubeModel, ShaderProgram * ShaderConverter) {
	if (!Map.IsValid()) return false;

	Material EquirectangularToCubemapMaterial = Material();
	EquirectangularToCubemapMaterial.SetShaderProgram(ShaderConverter);
	EquirectangularToCubemapMaterial.CullMode = Graphics::CM_None;

	Map.Use();
	{
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB16F, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB16F, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB16F, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB16F, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB16F, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB16F, Map.Width, Map.Width, 0, GL_RGB, GL_UNSIGNED_INT, NULL);

		Map.GenerateMipMaps();
	}

	RenderTarget Renderer = RenderTarget();

	static const Matrix4x4 CaptureProjection = Matrix4x4::Perspective(90.F * MathConstants::DegreeToRad, 1.F, 0.1F, 10.F);
	static const Matrix4x4 CaptureViews[] = {
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(-1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  1.F,  0.F), Vector3(0.F,  0.F,  1.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F, -1.F,  0.F), Vector3(0.F,  0.F, -1.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F,  1.F), Vector3(0.F, -1.F,  0.F)),
	   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F, -1.F), Vector3(0.F, -1.F,  0.F))
	};

	GLuint ModelMatrixBuffer;
	glGenBuffers(1, &ModelMatrixBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
	Renderer.SetUpBuffers();

	// --- Convert HDR equirectangular environment map to cubemap equivalent
	EquirectangularToCubemapMaterial.Use();
	EquirectangularToCubemapMaterial.SetTexture2D("_EquirectangularMap", Equirectangular, 0);
	EquirectangularToCubemapMaterial.SetMatrix4x4Array("_ProjectionMatrix", CaptureProjection.PointerToValue());
	EquirectangularToCubemapMaterial.CullMode = Graphics::CM_Front;

	const unsigned int MaxMipLevels = (unsigned int)Map.GetMipmapCount() + 1;
	for (unsigned int Lod = 0; Lod < MaxMipLevels; ++Lod) {
		// --- Reisze framebuffer according to mip-level size.
		unsigned int LodWidth = unsigned int(Map.Width) >> Lod;
		Renderer.Resize(LodWidth, LodWidth);
	
		float Roughness = (float)Lod / (float)(MaxMipLevels - 1);
		EquirectangularToCubemapMaterial.SetFloat1Array("_Roughness", &Roughness);
		for (unsigned int i = 0; i < 6; ++i) {
			EquirectangularToCubemapMaterial.SetMatrix4x4Array("_ViewMatrix", CaptureViews[i].PointerToValue());
	
			CubeModel->SetUpBuffers();
			CubeModel->BindVertexArray();
			EquirectangularToCubemapMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
			
			Renderer.PrepareTexture(&Map, i, Lod);
			Renderer.Clear();
			
			CubeModel->DrawElement();
			if (!Renderer.CheckStatus()) return false;
		}
	}
	Map.Deuse();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	Renderer.Delete();
	glDeleteBuffers(1, &ModelMatrixBuffer);
	return true;
}

void Cubemap::Delete() {
	glDeleteTextures(1, &TextureObject);
}
