
#include "CoreMinimal.h"
#include "Core/Layer.h"

namespace EmptySource {

	Layer::Layer(const WString & Name, unsigned int Level) :
		IIdentifier(Name), Name(Name), Level(Level) { 
	}

}