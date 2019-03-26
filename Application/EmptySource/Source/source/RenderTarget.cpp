#include "../include/RenderTarget.h"
#include "../include/CoreGraphics.h"
#include "../include/Utility/LogCore.h"

RenderTarget::RenderTarget() {
	Dimension = 0;
	TextureColor0Target = NULL;
	TextureDepthTarget = 0;
}

RenderTarget::RenderTarget(const IntVector2 & Size, Texture* Target, bool bUseDepth) {
	Dimension = Size;
	TextureColor0Target = Target;

	if (Target != NULL) {
		glGenFramebuffers(1, &FramebufferObject);
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferObject);

		// --- All future texture functions will modify this texture
		glBindTexture(GL_TEXTURE_2D, Target->GetTextureObject());

		// --- Set "RenderedTexture" as our colour attachement #0
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, Target->GetTextureObject(), 0);

		if (bUseDepth) {
			glGenRenderbuffers(1, &TextureDepthTarget);
			glBindRenderbuffer(GL_RENDERBUFFER, TextureDepthTarget);
			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, Size.x, Size.y);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, TextureDepthTarget);
		}

		// --- Set the list of draw buffers.
		GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
		glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

		// --- Always check that our framebuffer is ok
		glCheckFramebufferStatus(GL_FRAMEBUFFER);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

IntVector2 RenderTarget::GetDimension() const {
	return Dimension;
}

void RenderTarget::Use() const {
	glViewport(0, 0, Dimension.x, Dimension.y);
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferObject);
}

void RenderTarget::Clear() const {
	Use();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

bool RenderTarget::IsValid() const {
	return TextureColor0Target != NULL && TextureColor0Target->IsValid();
}

void RenderTarget::Delete() {
	glDeleteFramebuffers(1, &FramebufferObject);
}
