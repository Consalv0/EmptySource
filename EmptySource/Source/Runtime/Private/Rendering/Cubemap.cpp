
#include "Engine/Log.h"
#include "Engine/Core.h"
#include "Rendering/Cubemap.h"
#include "Rendering/GLFunctions.h"
#include "Rendering/Texture2D.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/Material.h"
#include "Mesh/Mesh.h"
#include "Mesh/MeshPrimitives.h"
#include "Math/Matrix4x4.h"
#include "Math/MathUtility.h"

namespace EmptySource {

	Cubemap::Cubemap(
		const int & WidthSize,
		const EColorFormat & Format,
		const EFilterMode & Filter,
		const ESamplerAddressMode & SamplerAddress)
	{
		Width = WidthSize;
		ColorFormat = Format;
		TextureObject = 0;

		if (WidthSize <= 0) {
			LOG_CORE_ERROR(L"One or more texture with incorrect size in cubemap with size {:d}", Width);
			return;
		}

		glGenTextures(1, &TextureObject);
		Use();
		SetFilterMode(Filter);
		SetSamplerAddressMode(SamplerAddress);
		Deuse();
	}

	void Cubemap::Use() const {
		if (IsValid()) {
			glBindTexture(GL_TEXTURE_CUBE_MAP, TextureObject);
		}
		else {
			LOG_CORE_ERROR(L"Texture Cubemap is not valid");
		}
	}

	void Cubemap::Deuse() const {
		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
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

	void Cubemap::SetFilterMode(const EFilterMode & Mode) {
		FilterMode = Mode;

		switch (Mode) {
		case FM_MinMagLinear:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, bLods ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
			break;
		case FM_MinMagNearest:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, bLods ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
			break;
		case FM_MinLinearMagNearest:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, bLods ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR);
			break;
		case FM_MinNearestMagLinear:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, bLods ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST);
			break;
		}
	}

	void Cubemap::SetSamplerAddressMode(const ESamplerAddressMode & Mode) {
		AddressMode = Mode;

		switch (Mode) {
		case SAM_Repeat:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT);
			break;
		case SAM_Mirror:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_MIRRORED_REPEAT);
			break;
		case SAM_Clamp:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			break;
		case SAM_Border:
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

	bool Cubemap::FromEquirectangular(Cubemap & Map, Texture2D * Equirectangular, ShaderProgram * ShaderConverter) {
		if (!Map.IsValid()) return false;

		Material EquirectangularToCubemapMaterial = Material();
		EquirectangularToCubemapMaterial.SetShaderProgram(ShaderConverter);
		EquirectangularToCubemapMaterial.CullMode = CM_None;

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
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(-1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F, -1.F,  0.F), Vector3(0.F,  0.F, -1.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F,  1.F,  0.F), Vector3(0.F,  0.F,  1.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F,  0.F,  1.F), Vector3(0.F, -1.F,  0.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F,  0.F, -1.F), Vector3(0.F, -1.F,  0.F))
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
		EquirectangularToCubemapMaterial.CullMode = CM_ClockWise;

		Renderer.Resize(Map.Width, Map.Width);
		for (unsigned int i = 0; i < 6; ++i) {
			EquirectangularToCubemapMaterial.SetMatrix4x4Array("_ViewMatrix", CaptureViews[i].PointerToValue());

			MeshPrimitives::Cube.SetUpBuffers();
			MeshPrimitives::Cube.BindVertexArray();
			EquirectangularToCubemapMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
			Renderer.PrepareTexture(&Map, i);
			Renderer.Clear();

			MeshPrimitives::Cube.DrawElement();
		}
		Map.GenerateMipMaps();
		Map.Deuse();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		Renderer.Delete();
		glDeleteBuffers(1, &ModelMatrixBuffer);
		return true;
	}

	bool Cubemap::FromHDREquirectangular(Cubemap & Map, Texture2D * Equirectangular, ShaderProgram * ShaderConverter) {
		if (!Map.IsValid()) return false;

		Material EquirectangularToCubemapMaterial = Material();
		EquirectangularToCubemapMaterial.SetShaderProgram(ShaderConverter);
		EquirectangularToCubemapMaterial.CullMode = CM_None;

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
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(-1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F, -1.F,  0.F), Vector3(0.F,  0.F, -1.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F,  1.F,  0.F), Vector3(0.F,  0.F,  1.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F,  0.F,  1.F), Vector3(0.F, -1.F,  0.F)),
		   Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F,  0.F, -1.F), Vector3(0.F, -1.F,  0.F))
		};

		GLuint ModelMatrixBuffer;
		glGenBuffers(1, &ModelMatrixBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
		Renderer.SetUpBuffers();

		// --- Convert HDR equirectangular environment map to cubemap equivalent
		EquirectangularToCubemapMaterial.Use();
		EquirectangularToCubemapMaterial.SetTexture2D("_EquirectangularMap", Equirectangular, 0);
		EquirectangularToCubemapMaterial.SetMatrix4x4Array("_ProjectionMatrix", CaptureProjection.PointerToValue());
		EquirectangularToCubemapMaterial.CullMode = CM_ClockWise;

		const unsigned int MaxMipLevels = (unsigned int)Map.GetMipmapCount() + 1;
		for (unsigned int Lod = 0; Lod < MaxMipLevels; ++Lod) {
			// --- Reisze framebuffer according to mip-level size.
			unsigned int LodWidth = (unsigned int)(Map.Width) >> Lod;
			Renderer.Resize(LodWidth, LodWidth);

			float Roughness = (float)Lod / (float)(MaxMipLevels - 1);
			EquirectangularToCubemapMaterial.SetFloat1Array("_Roughness", &Roughness);
			for (unsigned int i = 0; i < 6; ++i) {
				EquirectangularToCubemapMaterial.SetMatrix4x4Array("_ViewMatrix", CaptureViews[i].PointerToValue());

				MeshPrimitives::Cube.BindVertexArray();
				EquirectangularToCubemapMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);

				Renderer.PrepareTexture(&Map, i, Lod);
				Renderer.Clear();

				MeshPrimitives::Cube.DrawElement();
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

}