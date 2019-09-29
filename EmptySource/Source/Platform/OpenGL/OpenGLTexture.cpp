
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Shader.h"
#include "Rendering/Texture.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderingAPI.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"

#include "Rendering/MeshPrimitives.h"
#include "Math/Matrix4x4.h"
#include "Math/MathUtility.h"

#include "Platform/OpenGL/OpenGLDefinitions.h"
#include "Platform/OpenGL/OpenGLTexture.h"
#include "Platform/OpenGL/OpenGLRenderTarget.h"
#include "Platform/OpenGL/OpenGLAPI.h"

#include "Files/FileManager.h"

#include "glad/glad.h"

namespace ESource {

	OpenGLTexture2D::OpenGLTexture2D(
		const IntVector2 & Size,
		const EPixelFormat Format,
		const EFilterMode & Filter,
		const ESamplerAddressMode & Address)
	{

		glGenTextures(1, &TextureObject);

		Bind();
		SetFilterMode(Filter);
		SetSamplerAddressMode(Address);

		{
			glTexImage2D(
				GL_TEXTURE_2D, 0, OpenGLPixelFormatInfo[Format].InternalFormat, Size.x, Size.y, 0,
				OpenGLPixelFormatInfo[Format].InputFormat, OpenGLPixelFormatInfo[Format].BlockType, NULL
			);
			Unbind();
		}

		bValid = TextureObject != GL_FALSE && Size.MagnitudeSquared() > 0;
	}

	OpenGLTexture2D::OpenGLTexture2D(
		const IntVector2 & Size,
		const EPixelFormat Format,
		const EFilterMode & Filter,
		const ESamplerAddressMode & Address,
		const EPixelFormat InputFormat,
		const void * BufferData) 
	{
		glGenTextures(1, &TextureObject);
		bValid = TextureObject != GL_FALSE;
		Bind();
		SetFilterMode(Filter);
		SetSamplerAddressMode(Address);

		{
			glTexImage2D(
				GL_TEXTURE_2D, 0, OpenGLPixelFormatInfo[Format].InternalFormat, Size.x, Size.y, 0,
				OpenGLPixelFormatInfo[InputFormat].InputFormat, OpenGLPixelFormatInfo[InputFormat].BlockType, BufferData
			);
			Unbind();
		}
		
		bValid = TextureObject != GL_FALSE && Size.MagnitudeSquared() > 0;
	}

	OpenGLTexture2D::~OpenGLTexture2D() {
		LOG_CORE_DEBUG(L"Deleting texture2D '{}'...", TextureObject);
		glDeleteTextures(1, &TextureObject);
	}

	void OpenGLTexture2D::GenerateMipMaps(const EFilterMode & FilterMode, unsigned int Levels) {
		if (IsValid()) {
			Bind();
			SetFilterMode(FilterMode, true);
			glGenerateMipmap(GL_TEXTURE_2D);
			Unbind();
		}
	}

	void OpenGLTexture2D::Bind() const {
		if (!IsValid()) LOG_CORE_WARN(L"Texture 2D is not valid {}", TextureObject);
		glBindTexture(GL_TEXTURE_2D, TextureObject);
	}

	void OpenGLTexture2D::Unbind() const {
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	bool OpenGLTexture2D::IsValid() const {
		return bValid;
	}

	void OpenGLTexture2D::SetFilterMode(const EFilterMode & Mode, bool MipMaps) {
		switch (Mode) {
		case FM_MinMagLinear:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MipMaps ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
			break;
		case FM_MinMagNearest:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MipMaps ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
			break;
		case FM_MinLinearMagNearest:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MipMaps ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR);
			break;
		case FM_MinNearestMagLinear:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MipMaps ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST);
			break;
		}
	}

	void OpenGLTexture2D::SetSamplerAddressMode(const ESamplerAddressMode & Mode) {
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
		const EPixelFormat & Format,
		const EFilterMode & Filter,
		const ESamplerAddressMode & SamplerAddress)
	{
		bValid = false;

		if (WidthSize <= 0) {
			LOG_CORE_ERROR(L"One or more texture with incorrect size in cubemap with size {:d}", WidthSize);
			return;
		}

		glGenTextures(1, &TextureObject);
		bValid = TextureObject != GL_FALSE;
		Bind();
		SetFilterMode(Filter);
		SetSamplerAddressMode(SamplerAddress);
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, OpenGLPixelFormatInfo[Format].InternalFormat, WidthSize, WidthSize, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, OpenGLPixelFormatInfo[Format].InternalFormat, WidthSize, WidthSize, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, OpenGLPixelFormatInfo[Format].InternalFormat, WidthSize, WidthSize, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, OpenGLPixelFormatInfo[Format].InternalFormat, WidthSize, WidthSize, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, OpenGLPixelFormatInfo[Format].InternalFormat, WidthSize, WidthSize, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, OpenGLPixelFormatInfo[Format].InternalFormat, WidthSize, WidthSize, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
			Unbind(); 
		}

		bValid = TextureObject != GL_FALSE && WidthSize > 0;
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

	void OpenGLCubemap::GenerateMipMaps(const EFilterMode & FilterMode, unsigned int Levels) {
		if (IsValid()) {
			Bind();
			SetFilterMode(FilterMode, true);
			glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
			Unbind();
		}
	}

	void OpenGLCubemap::SetFilterMode(const EFilterMode & Mode, bool MipMaps) {
		switch (Mode) {
		case FM_MinMagLinear:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, MipMaps ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
			break;
		case FM_MinMagNearest:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, MipMaps ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
			break;
		case FM_MinLinearMagNearest:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, MipMaps ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR);
			break;
		case FM_MinNearestMagLinear:
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, MipMaps ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST);
			break;
		}
	}

	void OpenGLCubemap::SetSamplerAddressMode(const ESamplerAddressMode & Mode) {
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

	// bool OpenGLCubemap::ConvertFromCube(const CubeFaceTextures& Textures, bool bGenerateMipMaps) {
	// 	if (!IsValid()) return false;
	// 
	// 	if (!Textures.CheckDimensions(GetSize().x) || GetSize().x <= 0) {
	// 		return false;
	// 	}
	// 
	// 	Bind();
	// 	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x, 0,
	// 		GetOpenGLTextureColorFormatInput(Textures.Right.GetColorFormat()), GetOpenGLTextureColorFormat(Textures.Right.GetColorFormat()), Textures.Right.PointerToValue());
	// 	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x, 0,
	// 		GetOpenGLTextureColorFormatInput(Textures.Left.GetColorFormat()), GetOpenGLTextureColorFormat(Textures.Left.GetColorFormat()), Textures.Left.PointerToValue());
	// 	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x, 0,
	// 		GetOpenGLTextureColorFormatInput(Textures.Top.GetColorFormat()), GetOpenGLTextureColorFormat(Textures.Top.GetColorFormat()), Textures.Top.PointerToValue());
	// 	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x, 0,
	// 		GetOpenGLTextureColorFormatInput(Textures.Bottom.GetColorFormat()), GetOpenGLTextureColorFormat(Textures.Bottom.GetColorFormat()), Textures.Bottom.PointerToValue());
	// 	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x, 0,
	// 		GetOpenGLTextureColorFormatInput(Textures.Front.GetColorFormat()), GetOpenGLTextureColorFormat(Textures.Front.GetColorFormat()), Textures.Front.PointerToValue());
	// 	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GetOpenGLTextureColorFormat(GetColorFormat()), GetSize().x, GetSize().x, 0,
	// 		GetOpenGLTextureColorFormatInput(Textures.Back.GetColorFormat()), GetOpenGLTextureColorFormat(Textures.Back.GetColorFormat()), Textures.Back.PointerToValue());
	// 
	// 	if (bGenerateMipMaps) GenerateMipMaps();
	// 	Unbind();
	// 	return true;
	// }
	// 
	// bool OpenGLCubemap::ConvertFromHDREquirectangular(RTexturePtr Equirectangular, Material * EquirectangularToCubemapMaterial, bool bGenerateMipMaps) {
	// 	if (!IsValid()) return false;
	// 
	// 	Bind();
	// 	{
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB16F, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 
	// 		if (bGenerateMipMaps) GenerateMipMaps();
	// 	}
	// 
	// 	static const Matrix4x4 CaptureProjection = Matrix4x4::Perspective(90.F * MathConstants::DegreeToRad, 1.F, 0.1F, 10.F);
	// 	static const std::pair<ECubemapFace, Matrix4x4> CaptureViews[] = {
	// 	   { ECubemapFace::Right, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
	// 	   { ECubemapFace::Left,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(-1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
	// 	   { ECubemapFace::Up,    Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F, -1.F,  0.F), Vector3(0.F,  0.F, -1.F)) },
	// 	   { ECubemapFace::Down,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  1.F,  0.F), Vector3(0.F,  0.F,  1.F)) },
	// 	   { ECubemapFace::Back,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F,  1.F), Vector3(0.F, -1.F,  0.F)) },
	// 	   { ECubemapFace::Front, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F, -1.F), Vector3(0.F, -1.F,  0.F)) }
	// 	};
	// 
	// 	VertexBufferPtr ModelMatrixBuffer = VertexBuffer::Create(NULL, 0, EUsageMode::UM_Dynamic);
	// 	RenderTargetPtr Renderer = RenderTarget::Create();
	// 	
	// 	// --- Convert HDR equirectangular environment map to cubemap equivalent
	// 	EquirectangularToCubemapMaterial->Use();
	// 	EquirectangularToCubemapMaterial->SetTexture2D("_EquirectangularMap", Equirectangular, 0);
	// 	EquirectangularToCubemapMaterial->SetMatrix4x4Array("_ProjectionMatrix", CaptureProjection.PointerToValue());
	// 	
	// 	const unsigned int MaxMipLevels = (unsigned int)GetMipMapCount();
	// 	for (unsigned int Lod = 0; Lod <= MaxMipLevels; ++Lod) {
	// 
	// 		float Roughness = (float)Lod / (float)(MaxMipLevels);
	// 		EquirectangularToCubemapMaterial->SetFloat1Array("_Roughness", &Roughness);
	// 
	// 		Renderer->Bind();
	// 
	// 		for (auto View : CaptureViews) {
	// 			EquirectangularToCubemapMaterial->SetMatrix4x4Array("_ViewMatrix", View.second.PointerToValue());
	// 
	// 			MeshPrimitives::Cube.BindSubdivisionVertexArray(0);
	// 			EquirectangularToCubemapMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
	// 	
	// 			Renderer->BindCubemapFace(shared_from_this(), View.first, Lod);
	// 			Renderer->Clear();
	// 	
	// 			MeshPrimitives::Cube.DrawSubdivisionInstanciated(1, 0);
	// 			if (!Renderer->CheckStatus()) return false;
	// 		}
	// 	}
	// 	
	// 	Unbind();
	// 	return true;
	// }
	// 
	// bool OpenGLCubemap::ConvertFromEquirectangular(RTexturePtr Equirectangular, Material * EquirectangularToCubemapMaterial, bool bGenerateMipMaps) {
	// 	if (!IsValid()) return false;
	// 
	// 	Bind();
	// 	{
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB, GetSize().x, GetSize().y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	// 	}
	// 
	// 	static const Matrix4x4 CaptureProjection = Matrix4x4::Perspective(90.F * MathConstants::DegreeToRad, 1.F, 0.1F, 10.F);
	// 	static const std::pair<ECubemapFace, Matrix4x4> CaptureViews[] = {
	// 	   { ECubemapFace::Right, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
	// 	   { ECubemapFace::Left,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(-1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
	// 	   { ECubemapFace::Up,    Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F, -1.F,  0.F), Vector3(0.F,  0.F, -1.F)) },
	// 	   { ECubemapFace::Down,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F,  1.F,  0.F), Vector3(0.F,  0.F,  1.F)) },
	// 	   { ECubemapFace::Back,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F,  0.F,  1.F), Vector3(0.F, -1.F,  0.F)) },
	// 	   { ECubemapFace::Front, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(0.F,  0.F, -1.F), Vector3(0.F, -1.F,  0.F)) }
	// 	};
	// 
	// 	VertexBufferPtr ModelMatrixBuffer = VertexBuffer::Create(NULL, 0, EUsageMode::UM_Dynamic);
	// 	RenderTargetPtr Renderer = RenderTarget::Create();
	// 
	// 	// --- Convert HDR equirectangular environment map to cubemap equivalent
	// 	EquirectangularToCubemapMaterial->Use();
	// 	EquirectangularToCubemapMaterial->SetTexture2D("_EquirectangularMap", Equirectangular, 0);
	// 	EquirectangularToCubemapMaterial->SetMatrix4x4Array("_ProjectionMatrix", CaptureProjection.PointerToValue());
	// 
	// 	Renderer->Bind();
	// 	for (auto View : CaptureViews) {
	// 		EquirectangularToCubemapMaterial->SetMatrix4x4Array("_ViewMatrix", View.second.PointerToValue());
	// 
	// 		MeshPrimitives::Cube.SetUpBuffers();
	// 		MeshPrimitives::Cube.BindSubdivisionVertexArray(0);
	// 		EquirectangularToCubemapMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
	// 		Renderer->BindCubemapFace(shared_from_this(), View.first);
	// 		Renderer->Clear();
	// 
	// 		MeshPrimitives::Cube.DrawSubdivisionInstanciated(1, 0);
	// 	}
	// 	if (bGenerateMipMaps) GenerateMipMaps();
	// 	Unbind();
	// 	return true;
	// }

}
