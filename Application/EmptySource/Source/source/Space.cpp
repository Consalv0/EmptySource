
#include "../include/Core.h"
#include "../include/Space.h"
#include "../include/Object.h"

TDictionary<size_t, Space*> Space::AllSpaces = TDictionary<size_t, Space*>();

Space::Space() : IIdentifier(L"Space") {
	Name = GetIdentifierName();
}

Space::Space(const WString& InName) : IIdentifier(InName) {
	Name = InName;
	ObjectsIn = TDictionary<size_t, OObject*>();
}

Space::Space(Space & OtherSpace) : IIdentifier(OtherSpace.GetIdentifierName()) {
	WString Number, Residue;
	if (Text::GetLastNotOf(OtherSpace.Name, Residue, Number, L"0123456789"))
		Name = Residue + std::to_wstring(std::stoi(Number) + 1);
	else 
		Name = OtherSpace.Name + L"_1";
	ObjectsIn = TDictionary<size_t, OObject*>();
}

WString Space::GetName() {
	return Name;
}

Space * Space::GetMainSpace() {
	if (AllSpaces.size() == 0)
		return NULL;

	return AllSpaces.begin()->second;
}

Space * Space::GetSpace(const size_t & Identifier) {
	auto Find = AllSpaces.find(Identifier);
	if (AllSpaces.find(Identifier) == AllSpaces.end())
		return NULL;

	return Find->second;
}

Space * Space::CreateSpace(const WString & Name) {
	Space * NewSpace = new Space(Name);
	AllSpaces.insert(std::pair<const size_t, Space*>(NewSpace->GetIdentifierHash(), NewSpace));
	return NewSpace;
}

void Space::DeleteAllObjects() {
	for (TDictionary<size_t, OObject*>::iterator Iterator = ObjectsIn.begin(); Iterator != ObjectsIn.end(); Iterator++) {
		DeleteObject(Iterator->second);
	}
}

void Space::DeleteObject(OObject * Object) {
	Object->OnDelete();
	ObjectsIn.erase(Object->GetIdentifierHash());
	delete Object;
}

void Space::Destroy(Space * OtherSpace) {
	OtherSpace->DeleteAllObjects();
	AllSpaces.erase(OtherSpace->GetIdentifierHash());
	delete OtherSpace;
}

void Space::AddObject(OObject * Object) {
	ObjectsIn.insert(std::pair<const size_t, OObject*>(Object->GetIdentifierHash(), Object));
	Object->SpaceIn = this;
	Object->Initialize();
}
