
#include "../Source/include/Application.h"

using namespace EmptySource;

class SandboxApplication : public Application {
public:

	typedef Application Supper;
	SandboxApplication() : Supper() {}

};

Application * CreateInstance() {
	return new SandboxApplication();
}