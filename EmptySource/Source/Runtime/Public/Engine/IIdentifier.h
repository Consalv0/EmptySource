#pragma once

#include "CoreTypes.h"

namespace EmptySource {

	class IIdentifier {
	public:
		IIdentifier();
		IIdentifier(const WString & Name);

		WString GetUniqueName() const;
		size_t GetUniqueID() const;

	private:

		void ProcessIdentifierName(const WString & Name);

		WString InternalName;
		size_t NameHash;
	};

}