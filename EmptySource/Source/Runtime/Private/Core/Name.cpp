
#include "CoreMinimal.h"
#include "Utility/Hasher.h"
#include "Utility/TextFormatting.h"
#include "..\..\Public\Core\Name.h"

namespace ESource {

	TDictionary<size_t, WString> IName::NamesTable = TDictionary<size_t, WString>();
	TDictionary<size_t, size_t> IName::NameCountTable = TDictionary<size_t, size_t>();

	IName::IName(const WString & Text) : EntryName(Text) {
		ID = WStringToHash(Text);
		if (NamesTable.try_emplace(ID, Text).second) {
			NameCountTable.emplace(ID, 0);
			Number = 0;
		}
		else {
			Number = ++NameCountTable[ID];
		}
	}

	IName::IName(const WChar * Text) : EntryName(Text) {
		ID = WStringToHash(Text);
		if (NamesTable.try_emplace(ID, Text).second) {
			NameCountTable.emplace(ID, 0);
			Number = 0;
		}
		else {
			Number = ++NameCountTable[ID];
		}
	}

	IName::IName(size_t InNumber) {
		if (NamesTable.find(InNumber) == NamesTable.end()) {
			EntryName = L"";
			Number = 0;
		}
		else {
			EntryName = NamesTable[InNumber];
			Number = ++NameCountTable[InNumber];
		}
	}

	IName::IName(const WString & Text, size_t Number) : EntryName(Text), Number(Number) {
		ID = WStringToHash(Text);
		NamesTable.try_emplace(ID, Text);
	}

	IName::~IName() { }

	WString IName::GetDisplayName() const {
		return EntryName;
	}

	NString IName::GetNarrowDisplayName() const {
		return Text::WideToNarrow(EntryName);
	}

	WString IName::GetInstanceName() const {
		return EntryName + L"_" + std::to_wstring(Number);
	}

	NString IName::GetNarrowInstanceName() const {
		return Text::WideToNarrow(EntryName) + "_" + std::to_string(Number);
	}

	size_t IName::GetNumber() const {
		return Number;
	}

	size_t IName::GetInstanceID() const {
		return WStringToHash(EntryName + L"_" + std::to_wstring(Number));
	}

	size_t IName::GetID() const {
		return ID;
	}

	bool IName::operator<(const IName & Other) const {
		uint32_t i = 0;
		while ((i < EntryName.length()) && (i < Other.EntryName.length())) {
			if (tolower(EntryName[i]) < tolower(Other.EntryName[i])) return true;
			else if (tolower(EntryName[i]) > tolower(Other.EntryName[i])) return false;
			++i;
		}
		return (EntryName.length() + Number < Other.EntryName.length() + Other.Number);
	}

	bool IName::operator!=(const IName & Other) const {
		return ID != Other.ID || Number != Number;
	}

	bool IName::operator==(const IName & Other) const {
		return ID == Other.ID && Number == Number;
	}

}
