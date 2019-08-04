
#include "Engine/EmptySource.h"

class SandboxApplication : public EmptySource::Application {
public:
	typedef Application Supper;
	
	SandboxApplication() : Supper() {}

};

EmptySource::Application * EmptySource::CreateApplication() {
	return new SandboxApplication();
}
