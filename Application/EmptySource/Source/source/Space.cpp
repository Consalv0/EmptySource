
#include "../include/Core.h"
#include "../include/Space.h"
#include "../include/EmptyObject.h"

TDictionary<size_t, Space*> Space::AllSpaces = TDictionary<size_t, Space*>();

Space::Space() : IIdentifier() {
	InternalName = L"Space_" + std::to_wstring(GetIdentifier());
	Name = L"Space";
}

Space::Space(const WString& InName) : IIdentifier() {
	InternalName = InName;
	if (!Text::ReplaceUntil(InternalName, L"_", L"_" + std::to_wstring(GetIdentifier()))) {
		InternalName += L"_" + std::to_wstring(GetIdentifier());
	}
	Name = InName;
	ObjectsIn = TDictionary<size_t, EmptyObject*>();
}

Space::Space(Space & OtherSpace) : IIdentifier() {
	InternalName = OtherSpace.InternalName;
	if (!Text::ReplaceUntil(InternalName, L"_", L"_" + std::to_wstring(GetIdentifier()))) {
		InternalName += L"_" + std::to_wstring(GetIdentifier());
	}
	WString Number, Residue;
	if (Text::GetLastNotOf(OtherSpace.Name, Residue, Number, L"0123456789"))
		Name = Residue + std::to_wstring(std::stoi(Number) + 1);
	else 
		Name = OtherSpace.Name + L"_1";
	ObjectsIn = TDictionary<size_t, EmptyObject*>();
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
	AllSpaces.insert(std::pair<const size_t, Space*>(NewSpace->GetIdentifier(), NewSpace));
	return NewSpace;
}

void Space::DestroyAllObjects() {
	for (TDictionary<size_t, EmptyObject*>::iterator Iterator = ObjectsIn.begin(); Iterator != ObjectsIn.end(); Iterator++) {
		Iterator->second->Delete();
	}
}

void Space::DestroyObject(EmptyObject * object) {
	ObjectsIn.erase(object->GetIdentifier());
	delete object;
}

void Space::Destroy(Space * OtherSpace) {
	OtherSpace->DestroyAllObjects();
	AllSpaces.erase(OtherSpace->GetIdentifier());
	delete OtherSpace;
}

void Space::Add(EmptyObject * object) {
	ObjectsIn.insert(std::pair<const size_t, EmptyObject*>(object->GetIdentifier(), object));
	object->SpaceIn = this;
	object->Initialize();
}
