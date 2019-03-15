
#include "..\External\ft2build.h"
#include FT_FREETYPE_H
#include FT_OUTLINE_H
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
	Face = NULL;
}

void Font::Clear() {
	FT_Done_Face(Face);
}

void Font::SetGlyphHeight(const unsigned int & Size) const {
	FT_Error Error = FT_Set_Pixel_Sizes(Face, 0, Size);
	if (Error) {
		Debug::Log(Debug::LogError, L"Couldn't set the glyph size to %d, %s", Size, FT_ErrorMessage(Error));
	}
}

Point2 FT_Point2(const FT_Vector & Vector) {
	return Point2(Vector.x / 64.F, Vector.y / 64.F);
}

int FT_MoveTo(const FT_Vector * To, void * User) {
	FT_Context *Context = reinterpret_cast<FT_Context *>(User);
	Context->contour = &Context->shape->addContour();
	Context->position = FT_Point2(*To);
	return 0;
}

int FT_LineTo(const FT_Vector * To, void * User) {
	FT_Context *Context = reinterpret_cast<FT_Context *>(User);
	Context->contour->addEdge(new LinearSegment(Context->position, FT_Point2(*To)));
	Context->position = FT_Point2(*To);
	return 0;
}

int FT_ConicTo(const FT_Vector * Control, const FT_Vector * To, void * User) {
	FT_Context *Context = reinterpret_cast<FT_Context *>(User);
	Context->contour->addEdge(new QuadraticSegment(Context->position, FT_Point2(*Control), FT_Point2(*To)));
	Context->position = FT_Point2(*To);
	return 0;
}

int FT_CubicTo(const FT_Vector * ControlA, const FT_Vector * ControlB, const FT_Vector * To, void * User) {
	FT_Context *Context = reinterpret_cast<FT_Context *>(User);
	Context->contour->addEdge(new CubicSegment(Context->position, FT_Point2(*ControlA), FT_Point2(*ControlB), FT_Point2(*To)));
	Context->position = FT_Point2(*To);
	return 0;
}

void Font::Initialize(FileStream * File) {
	FT_Error Error;
	if (Error = FT_New_Face(FreeTypeLibrary, WStringToString(File->GetPath()).c_str(), 0, &Face))
		Debug::Log(Debug::LogError, L"Failed to load font, %s", FT_ErrorMessage(Error));
}

unsigned int Font::GetGlyphIndex(const unsigned long & Character) const {
	return FT_Get_Char_Index(Face, Character);
}

bool Font::GetGlyph(FontGlyph & Glyph, const unsigned int& Character) {
	FT_Error Error = FT_Load_Glyph(Face, GetGlyphIndex(Character), FT_LOAD_NO_SCALE);
	if (Error) {
		Debug::Log(Debug::LogError, L"Failed to load Glyph '%c', %s", Character, FT_ErrorMessage(Error));
		return false;
	}
	FT_GlyphSlot & FTGlyph = Face->glyph;

	Glyph.UnicodeValue = Character;
	Glyph.VectorShape.contours.clear();
	Glyph.VectorShape.inverseYAxis = false;
	Glyph.Advance = FTGlyph->advance.x / 64.F;

	FT_Context Context = { };
	Context.shape = &Glyph.VectorShape;
	FT_Outline_Funcs FT_Functions;
	FT_Functions.move_to = &FT_MoveTo;
	FT_Functions.line_to = &FT_LineTo;
	FT_Functions.conic_to = &FT_ConicTo;
	FT_Functions.cubic_to = &FT_CubicTo;
	FT_Functions.shift = 0;
	FT_Functions.delta = 0;
	Error = FT_Outline_Decompose(&FTGlyph->outline, &FT_Functions, &Context);
	if (Error) {
		Debug::Log(Debug::LogError, L"Failed to decompose outline of Glyph '%c', %s", Character, FT_ErrorMessage(Error));
		return false;
	}
	return true;
}
