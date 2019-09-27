#pragma once

namespace ESource {

	class IName {
	public:
		IName(const WString & Text);

		IName(const WChar * Text);

		IName(size_t Number);

		IName(const WString & Text, size_t Number);

		~IName();

		WString GetDisplayName() const;

		NString GetNarrowDisplayName() const;

		WString GetInstanceName() const;

		NString GetNarrowInstanceName() const;
		
		size_t GetNumber() const;

		size_t GetInstanceID() const;
		
		size_t GetID() const;

		bool operator<(const IName& Other) const;

		bool operator==(const IName& Other) const;

		bool operator!=(const IName& Other) const;

	private:
		size_t Number;
		WString EntryName;
		size_t ID;

		static TDictionary<size_t, WString> NamesTable;
		static TDictionary<size_t, size_t> NameCountTable;
	};

}