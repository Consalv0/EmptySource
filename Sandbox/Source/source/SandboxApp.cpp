
#include "../Source/include/Application.h"

class SandboxApplication : public Application {
public:

	typedef Application Supper;
	SandboxApplication() : Supper() {}

};

Application * CreateInstance() {
	return new SandboxApplication();
}