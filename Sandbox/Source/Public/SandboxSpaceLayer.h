
#include "Core/SpaceLayer.h"

class SandboxSpaceLayer : public EmptySource::SpaceLayer {
protected:
	typedef SpaceLayer Super;
public:
	SandboxSpaceLayer(const EmptySource::WString & Name, unsigned int Level);

	void OnAwake() override;

	void OnImGuiRender() override;

};