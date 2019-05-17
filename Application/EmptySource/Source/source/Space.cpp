
#include "../include/Core.h"
#include "../include/Space.h"
#include "../include/Object.h"

TDictionary<size_t, Space*> Space::AllSpaces = TDictionary<size_t, Space*>();

Space::Space() : IIdentifier(L"Space") {
	Name = GetIdentifierName();
}

Space::Space(const WString& InName) : IIdentifier(InName) {
	Name = InName;
	ObjectsIn = TDictionary<size_t, Object*>();
}

Space::Space(Space & OtherSpace) : IIdentifier(OtherSpace.GetIdentifierName()) {
	WString Number, Residue;
	if (Text::GetLastNotOf(OtherSpace.Name, Residue, Number, L"0123456789"))
		Name = Residue + std::to_wstring(std::stoi(Number) + 1);
	else 
		Name = OtherSpace.Name + L"_1";
	ObjectsIn = TDictionary<size_t, Object*>();
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

void Space::DestroyAllObjects() {
	for (TDictionary<size_t, Object*>::iterator Iterator = ObjectsIn.begin(); Iterator != ObjectsIn.end(); Iterator++) {
		Iterator->second->Delete();
	}
}

void Space::DestroyObject(Object * object) {
	ObjectsIn.erase(object->GetIdentifierHash());
	delete object;
}

void Space::Destroy(Space * OtherSpace) {
	OtherSpace->DestroyAllObjects();
	AllSpaces.erase(OtherSpace->GetIdentifierHash());
	delete OtherSpace;
}

void Space::Add(Object * object) {
	ObjectsIn.insert(std::pair<const size_t, Object*>(object->GetIdentifierHash(), object));
	object->SpaceIn = this;
	object->Initialize();
}
