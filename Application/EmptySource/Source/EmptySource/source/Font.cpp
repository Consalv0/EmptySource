
#include "..\External\ft2build.h"
#include FT_FREETYPE_H
#include "..\External\freetype\freetype.h"

#include "..\include\Core.h"
#include "..\include\Utility\LogFreeType.h"
#include "..\include\Utility\LogCore.h"
#include "..\include\FileStream.h"
#include "..\include\Font.h"

FT_LibraryRec_ * Font::FreeTypeLibrary = NULL;

bool Font::InitializeFreeType() {
	// --- Free Type 2 Library is already instanciated
	if (FreeTypeLibrary != NULL) {
		return true;
	}

	FT_Error Error;
	if (Error = FT_Init_FreeType(&FreeTypeLibrary)) {
		Debug::Log(Debug::LogCritical, L"Could not initialize FreeType Library, %s", FT_ErrorMessage(Error));
		return false;
	}

	return true;
}

Font::Font() {
	FreeTypeFace = NULL;
}

void Font::Clear() {
	FT_Done_Face(FreeTypeFace);
}

void Font::SetGlyphHeight(const unsigned int & Size) const {
	FT_Error Error = FT_Set_Pixel_Sizes(FreeTypeFace, 0, Size);
	if (Error) {
		Debug::Log(Debug::LogError, L"Couldn't set the glyph size to %d, %s", Size, FT_ErrorMessage(Error));
	}
}

void Font::Initialize(FileStream * File) {
	FT_Error Error;
	if (Error = FT_New_Face(FreeTypeLibrary, WStringToString(File->GetPath()).c_str(), 0, &FreeTypeFace))
		Debug::Log(Debug::LogError, L"Failed to load font, %s", FT_ErrorMessage(Error));
}

unsigned int Font::GetGlyphIndex(const unsigned long & Character) const {
	return FT_Get_Char_Index(FreeTypeFace, Character);
}
