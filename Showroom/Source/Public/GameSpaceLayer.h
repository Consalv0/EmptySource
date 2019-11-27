
#include "Core/SpaceLayer.h"

class GameSpaceLayer : public ESource::SpaceLayer {
protected:
	typedef SpaceLayer Super;
public:
	GameSpaceLayer(const ESource::WString & Name, unsigned int Level);

	void OnAwake() override;

	void OnRender() override;

	void OnPostRender() override;

	void OnImGuiRender() override;

};