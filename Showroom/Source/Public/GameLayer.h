
#include "Core/SpaceLayer.h"

class SandboxSpaceLayer : public ESource::SpaceLayer {
protected:
	typedef SpaceLayer Super;
public:
	SandboxSpaceLayer(const ESource::WString & Name, unsigned int Level);

	void OnAwake() override;

	void OnRender() override;

	void OnImGuiRender() override;

};