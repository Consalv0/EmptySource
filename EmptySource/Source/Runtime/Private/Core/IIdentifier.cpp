
#include "CoreMinimal.h"
#include "Utility/TextFormatting.h"
#include "Core/IIdentifier.h"
#include "Utility/Hasher.h"
#include <stddef.h>

namespace EmptySource {

	size_t IIdentifier::GetUniqueID() const {
		return NameHash;
	}

	void IIdentifier::ProcessIdentifierName(const WString & Name) {
		// Identifier Generator
		static size_t CurrentIdentifier;

		InternalName = Name;
		if (!Text::ReplaceFromLast(InternalName, L"_", L"_" + std::to_wstring(++CurrentIdentifier))) {
			InternalName += L"_" + std::to_wstring(CurrentIdentifier);
		}
		NameHash = WStringToHash(InternalName);
	}

	WString IIdentifier::GetUniqueName() const {
		return InternalName;
	}

	IIdentifier::IIdentifier() {
		ProcessIdentifierName(L"");
	}

	IIdentifier::IIdentifier(const WString & Name) {
		ProcessIdentifierName(Name);
	}

}