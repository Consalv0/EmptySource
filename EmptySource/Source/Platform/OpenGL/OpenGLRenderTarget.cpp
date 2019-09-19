
#include "CoreMinimal.h"
#include "Rendering/Rendering.h"
#include "Resources/ShaderProgram.h"
#include "Rendering/Texture.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderingAPI.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"

#include "Rendering/MeshPrimitives.h"
#include "Math/Matrix4x4.h"
#include "Math/MathUtility.h"

#include "Platform/OpenGL/OpenGLTexture.h"
#include "Platform/OpenGL/OpenGLRenderTarget.h"
#include "Platform/OpenGL/OpenGLAPI.h"

#include "glad/glad.h"

namespace EmptySource {
	
	unsigned int GetOpenGLCubemapFace(const ECubemapFace & CF) {
		switch (CF) {
			default:
			case ECubemapFace::Right:
				return GL_TEXTURE_CUBE_MAP_POSITIVE_X;
			case ECubemapFace::Left:
				return GL_TEXTURE_CUBE_MAP_NEGATIVE_X;
			case ECubemapFace::Up:
				return GL_TEXTURE_CUBE_MAP_POSITIVE_Y;
			case ECubemapFace::Down:
				return GL_TEXTURE_CUBE_MAP_NEGATIVE_Y;
			case ECubemapFace::Back:
				return GL_TEXTURE_CUBE_MAP_POSITIVE_Z;
			case ECubemapFace::Front:
				return GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
		}
	}

	OpenGLRenderTarget::OpenGLRenderTarget()
		: RenderingTexture(NULL), Dimension(ETextureDimension::None) {
		glGenFramebuffers(1, &FramebufferObject);
		glGenRenderbuffers(1, &RenderbufferObject);
	}

	OpenGLRenderTarget::~OpenGLRenderTarget() {
		ReleaseTexture(); Unbind();
		glDeleteFramebuffers(1, &FramebufferObject);
		glDeleteRenderbuffers(1, &RenderbufferObject);
	}

	void OpenGLRenderTarget::Bind() const {
		if (!IsValid()) LOG_CORE_ERROR(L"RenderTarget is not valid");
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferObject);
		glBindRenderbuffer(GL_RENDERBUFFER, RenderbufferObject);
	}

	void OpenGLRenderTarget::Unbind() const {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
	}

	bool OpenGLRenderTarget::CheckStatus() const {
		const GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		return Status == GL_FRAMEBUFFER_COMPLETE;
	}

	void * OpenGLRenderTarget::GetNativeObject() const {
		if (IsValid()) return (void *)(unsigned long long) FramebufferObject;
		return 0;
	}

	TexturePtr OpenGLRenderTarget::GetBindedTexture() const {
		return RenderingTexture;
	}

	void OpenGLRenderTarget::BindTexture2D(const TexturePtr & Texture, int Lod, int TextureAttachment) {
		RenderingTexture = Texture;
		Size = RenderingTexture->GetSize();
		if (!IsValid()) return;
		Bind();

		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, Size.x, Size.y);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RenderbufferObject);

		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + TextureAttachment, (GLuint)(unsigned long long)RenderingTexture->GetTextureObject(), Lod);
		// Set the list of draw buffers.
		GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 + (GLenum)TextureAttachment };
		glDrawBuffers(1, DrawBuffers);

		glViewport(0, 0, Size.x, Size.y);
	}

	void OpenGLRenderTarget::BindCubemapFace(const TexturePtr & Texture, ECubemapFace Face, int Lod, int TextureAttachment) {
		RenderingTexture = Texture;
		Size = RenderingTexture->GetSize();
		if (!IsValid()) return;
		Bind();

		unsigned int LodWidth = (unsigned int)(Texture->GetSize().x) >> Lod;
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, LodWidth, LodWidth);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RenderbufferObject);
		
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + TextureAttachment,
			GetOpenGLCubemapFace(Face), (GLuint)(unsigned long long)RenderingTexture->GetTextureObject(), Lod);

		glViewport(0, 0, LodWidth, LodWidth);
	}

	void OpenGLRenderTarget::ReleaseTexture() {
		RenderingTexture.reset();
	}

	void OpenGLRenderTarget::Resize(const IntVector3 & NewSize) {
		Size = { Math::Min(0, NewSize.x), Math::Min(0, NewSize.y), Math::Min(0, NewSize.z) };
	}

	void OpenGLRenderTarget::Clear() const {
		Rendering::ClearCurrentRender(true, 0, true, 1, false, 0);
	}

	bool OpenGLRenderTarget::IsValid() const {
		return FramebufferObject != GL_FALSE && RenderbufferObject != GL_FALSE;
	}

}