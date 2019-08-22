
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Shader.h"
#include "Rendering/Texture.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderingAPI.h"

#include "Rendering/Material.h"
#include "Mesh/Mesh.h"
#include "Mesh/MeshPrimitives.h"
#include "Math/Matrix4x4.h"
#include "Math/MathUtility.h"

#include "Platform/OpenGL/OpenGLTexture.h"
#include "Platform/OpenGL/OpenGLRenderTarget.h"
#include "Platform/OpenGL/OpenGLAPI.h"

#include "Files/FileManager.h"

#include "glad/glad.h"

namespace EmptySource {
	
	unsigned int GetOpenGLTextureColorFormat(const EColorFormat & CF) {
		switch (CF) {
			case CF_Red:
				return GL_RED;
			case CF_RG:
				return GL_RG;
			case CF_RGB:
				return GL_RGB;
			case CF_RGBA:
				return GL_RGBA;
			case CF_RG16F:
				return GL_RG16F;
			case CF_RGB16F:
				return GL_RGB16F;
			case CF_RGBA16F:
				return GL_RGBA16F;
			case CF_RGBA32F:
				return GL_RGBA32F;
			case CF_RGB32F:
				return GL_RGB32F;
			default:
				LOG_CORE_WARN(L"Color not implemented, using RGBA");
				return GL_RGBA;
		}
	}

	unsigned int GetOpenGLTextureColorFormatInput(const EColorFormat & CF) {
		switch (CF) {
			case CF_Red:
				return GL_RED;
			case CF_RG16F:
			case CF_RG:
				return GL_RG;
			case CF_RGB32F:
			case CF_RGB16F:
			case CF_RGB:
				return GL_RGB;
			case CF_RGBA32F:
			case CF_RGBA16F:
			case CF_RGBA:
				return GL_RGBA;
			default:
				LOG_CORE_WARN(L"Color not implemented, using RGBA");
				return GL_RGBA;
		}
	}

	unsigned int GetOpenGLTextureInputType(const EColorFormat & CF) {
		switch (CF) {
			case CF_Red:
			case CF_RG:
			case CF_RGB:
			case CF_RGBA:
				return GL_UNSIGNED_BYTE;
			case CF_RG16F:
			case CF_RGB16F:
			case CF_RGBA16F:
			case CF_RGBA32F:
			case CF_RGB32F:
				return GL_FLOAT;
			default:
				LOG_CORE_WARN(L"Color not implemented, using unsigned byte");
				return GL_UNSIGNED_BYTE;
		}
	}

	OpenGLTexture2D::OpenGLTexture2D(
		const IntVector2 & Size,
		const EColorFormat Format,
		const EFilterMode & Filter,
		const ESamplerAddressMode & Address)
		: Size(Size), ColorFormat(Format), MipMapCount(1), TextureObject(0)
	{

		glGenTextures(1, &TextureObject);

		Bind();
		SetFilterMode(Filter);
		SetSamplerAddressMode(Address);

		{
			glTexImage2D(
				GL_TEXTURE_2D, 0, GetOpenGLTextureColorFormat(ColorFormat), Size.x, Size.y, 0,
				GL_RGBA, GL_FLOAT, NULL
			);
			Unbind();
		}

		bValid = TextureObject != GL_FALSE && GetSize().MagnitudeSquared() > 0;
	}

	OpenGLTexture2D::OpenGLTexture2D(
		const IntVector2 & Size,
		const EColorFormat Format,
		const EFilterMode & Filter,
		const ESamplerAddressMode & Address,
		const EColorFormat InputFormat,
		const void * BufferData) 
		: Size(Size), ColorFormat(Format), MipMapCount(1)
	{
		glGenTextures(1, &TextureObject);
		bValid = TextureObject != GL_FALSE;
		Bind();
		SetFilterMode(Filter);
		SetSamplerAddressMode(Address);

		{
			glTexImage2D(
				GL_TEXTURE_2D, 0, GetOpenGLTextureColorFormat(ColorFormat), Size.x, Size.y, 0,
				GetOpenGLTextureColorFormatInput(InputFormat), GetOpenGLTextureInputType(InputFormat), BufferData
			);
			Unbind();
		}
		
		bValid = TextureObject != GL_FALSE && GetSize().MagnitudeSquared() > 0;
	}

	OpenGLTexture2D::~OpenGLTexture2D() {
		LOG_CORE_DEBUG(L"Deleting texture2D '{}'...", TextureObject);
		glDeleteTextures(1, &TextureObject);
	}

	void OpenGLTexture2D::GenerateMipMaps() {
		if (IsValid()) {
			MipMapCount = (int)log2f((float)GetWidth());
			Bind();
			SetFilterMode(FilterMode);
			glGenerateMipmap(GL_TEXTURE_2D);
			Unbind();
		}
	}

	void OpenGLTexture2D::Bind() const {
		if (!IsValid()) LOG_CORE_WARN(L"Texture 2D is not valid");
		glBindTexture(GL_TEXTURE_2D, TextureObject);
	}

	void OpenGLTexture2D::Unbind() const {
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	bool OpenGLTexture2D::IsValid() const {
		return bValid;
	}

	void OpenGLTexture2D::SetFilterMode(const EFilterMode & Mode) {
		FilterMode = Mode;

		switch (Mode) {
		case FM_MinMagLinear:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MipMapCount > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
			break;
		case FM_MinMagNearest:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MipMapCount > 1 ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
			break;
		case FM_MinLinearMagNearest:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MipMapCount > 1 ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR);
			break;
		case FM_MinNearestMagLinear:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MipMapCount > 1 ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST);
			break;
		}
	}

	void OpenGLTexture2D::SetSamplerAddressMode(const ESamplerAddressMode & Mode) {
		AddressMode = Mode;

		switch (Mode) {
		case SAM_Repeat:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			break;
		case SAM_Mirror:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
			break;
		case SAM_Clamp:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			break;
		case SAM_Border:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			break;
		}
	}

	// Cubemap

	OpenGLCubemap::OpenGLCubemap(
		const int & WidthSize,
		const EColorFormat & Format,
		const EFilterMode & Filter,
		const ESamplerAddressMode & SamplerAddress)
	{
		Size = WidthSize;
		ColorFormat = Format;
		MipMapCount = 1;
		bValid = false;

		if (WidthSize <= 0) {
			LOG_CORE_ERROR(L"One or more texture with incorrect size in cubemap with size {:d}", Size);
			return;
		}

		glGenTextures(1, &TextureObject);
		bValid = TextureObject != GL_FALSE;
		Bind();
		SetFilterMode(Filter);
		SetSamplerAddressMode(SamplerAddress);
		Unbind();

		bValid = TextureObject != GL_FALSE && Size > 0;
	}

	OpenGLCubemap::~OpenGLCubemap() {
		LOG_CORE_DEBUG(L"Deleting cubemap texture '{}'...", TextureObject);
		glDeleteTextures(1, &TextureObject);
	}

	void OpenGLCubemap::Bind() const {
		if (!IsValid()) LOG_CORE_ERROR(L"Texture Cubemap is not valid");
		glBindTexture(GL_TEXTURE_CUBE_MAP, TextureObject);
	}

	void OpenGLCubemap::Unbind() const {
		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	}

	bool OpenGLCubemap::IsValid() const {
		return bValid;
	}

	void OpenGLCubemap::GenerateMipMaps() {
		if (IsValid()) {
			MipMapCount = (int)log2f((float)GetSize().x);
			Bind();
			SetFilterMode(FilterMode);
			glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
			Unbind();
		}
	}

	void OpenGLCubemap::SetFilterMode(const EFilterMode & Mode) {
		FilterMode = Mode;

		switch (Mode) {
		case FM_MinMagLinear:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, MipMapCount > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
			break;
		case FM_MinMagNearest:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, MipMapCount > 1 ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
			break;
		case FM_MinLinearMagNearest:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, MipMapCount > 1 ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR);
			break;
		case FM_MinNearestMagLinear:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, MipMapCount > 1 ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST);
			break;
		}
	}

	void OpenGLCubemap::SetSamplerAddressMode(const ESamplerAddressMode & Mode) {
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

	bool OpenGLCubemap::ConvertFromCube(const CubeFaceTextures<UCharRGB>& Textures, bool bGenerateMipMaps) {
		if (!IsValid()) return false;

		if (!Textures.CheckDimensions(GetSize().x) || GetSize().x <= 0) {
			return false;
		}

		Bind();
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Right.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Left.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Top.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Bottom.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Front.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_UNSIGNED_BYTE, Textures.Back.PointerToValue());

		if (bGenerateMipMaps) GenerateMipMaps();
		Unbind();
		return true;
	}

	bool OpenGLCubemap::ConvertFromHDRCube(const CubeFaceTextures<FloatRGB>& Textures, bool bGenerateMipMaps) {
		if (!IsValid()) return false;

		if (!Textures.CheckDimensions(GetSize().x) || GetSize().x <= 0) {
			return false;
		}

		Bind();
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_FLOAT, Textures.Right.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_FLOAT, Textures.Left.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_FLOAT, Textures.Top.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_FLOAT, Textures.Bottom.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_FLOAT, Textures.Front.PointerToValue());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x,
			0, GL_RGB, GL_FLOAT, Textures.Back.PointerToValue());

		if (bGenerateMipMaps) GenerateMipMaps();
		Unbind();
		return true;
	}

	bool OpenGLCubemap::ConvertFromEquirectangular(Texture2DPtr Equirectangular, Material * EquirectangularToCubemapMaterial, bool bGenerateMipMaps) {
		if (!IsValid()) return false;

		Bind();
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
		}

		static const Matrix4x4 CaptureProjection = Matrix4x4::Perspective(90.F * MathConstants::DegreeToRad, 1.F, 0.1F, 10.F);
		static const std::pair<ECubemapFace, Matrix4x4> CaptureViews[] = {
		   { ECubemapFace::Right, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
		   { ECubemapFace::Left,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(-1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
		   { ECubemapFace::Up,    Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F, -1.F,  0.F), Vector3(0.F,  0.F, -1.F)) },
		   { ECubemapFace::Down,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  1.F,  0.F), Vector3(0.F,  0.F,  1.F)) },
		   { ECubemapFace::Back,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F,  1.F), Vector3(0.F, -1.F,  0.F)) },
		   { ECubemapFace::Front, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F, -1.F), Vector3(0.F, -1.F,  0.F)) }
		};

		GLuint ModelMatrixBuffer;
		glGenBuffers(1, &ModelMatrixBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
		RenderTargetPtr Renderer = RenderTarget::Create();
		
		// --- Convert HDR equirectangular environment map to cubemap equivalent
		EquirectangularToCubemapMaterial->Use();
		EquirectangularToCubemapMaterial->SetTexture2D("_EquirectangularMap", Equirectangular, 0);
		EquirectangularToCubemapMaterial->SetMatrix4x4Array("_ProjectionMatrix", CaptureProjection.PointerToValue());
		
		Renderer->Bind();
		for (auto View : CaptureViews) {
			EquirectangularToCubemapMaterial->SetMatrix4x4Array("_ViewMatrix", View.second.PointerToValue());
		
			MeshPrimitives::Cube.SetUpBuffers();
			MeshPrimitives::Cube.BindVertexArray();
			EquirectangularToCubemapMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
			Renderer->BindCubemapFace(shared_from_this(), View.first);
			Renderer->Clear();
		
			MeshPrimitives::Cube.DrawElement();
		}
		if (bGenerateMipMaps) GenerateMipMaps();
		Unbind();
		
		glDeleteBuffers(1, &ModelMatrixBuffer);
		return true;
	}

	bool OpenGLCubemap::ConvertFromHDREquirectangular(Texture2DPtr Equirectangular, Material * EquirectangularToCubemapMaterial, bool bGenerateMipMaps) {
		if (!IsValid()) return false;

		Bind();
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);

			if (bGenerateMipMaps) GenerateMipMaps();
		}

		static const Matrix4x4 CaptureProjection = Matrix4x4::Perspective(90.F * MathConstants::DegreeToRad, 1.F, 0.1F, 10.F);
		static const std::pair<ECubemapFace, Matrix4x4> CaptureViews[] = {
		   { ECubemapFace::Right, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
		   { ECubemapFace::Left,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(-1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
		   { ECubemapFace::Up,    Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F, -1.F,  0.F), Vector3(0.F,  0.F, -1.F)) },
		   { ECubemapFace::Down,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  1.F,  0.F), Vector3(0.F,  0.F,  1.F)) },
		   { ECubemapFace::Back,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F,  1.F), Vector3(0.F, -1.F,  0.F)) },
		   { ECubemapFace::Front, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F, -1.F), Vector3(0.F, -1.F,  0.F)) }
		};

		GLuint ModelMatrixBuffer;
		glGenBuffers(1, &ModelMatrixBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
		RenderTargetPtr Renderer = RenderTarget::Create();
		
		// --- Convert HDR equirectangular environment map to cubemap equivalent
		EquirectangularToCubemapMaterial->Use();
		EquirectangularToCubemapMaterial->SetTexture2D("_EquirectangularMap", Equirectangular, 0);
		EquirectangularToCubemapMaterial->SetMatrix4x4Array("_ProjectionMatrix", CaptureProjection.PointerToValue());
		
		const unsigned int MaxMipLevels = (unsigned int)GetMipMapCount();
		for (unsigned int Lod = 0; Lod <= MaxMipLevels; ++Lod) {

			float Roughness = (float)Lod / (float)(MaxMipLevels);
			EquirectangularToCubemapMaterial->SetFloat1Array("_Roughness", &Roughness);

			Renderer->Bind();

			for (auto View : CaptureViews) {
				EquirectangularToCubemapMaterial->SetMatrix4x4Array("_ViewMatrix", View.second.PointerToValue());

				MeshPrimitives::Cube.BindVertexArray();
				EquirectangularToCubemapMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
		
				Renderer->BindCubemapFace(shared_from_this(), View.first, Lod);
				Renderer->Clear();
		
				MeshPrimitives::Cube.DrawElement();
				if (!Renderer->CheckStatus()) return false;
			}
		}
		
		Unbind();
		glDeleteBuffers(1, &ModelMatrixBuffer);
		return true;
	}

}
