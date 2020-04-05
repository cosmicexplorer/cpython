#include <Block.h>
#include <stdio.h>

#include "clang-c/Index.h"

int main(int argc, char **argv) {
    CXCursorVisitorBlock block = ^(CXCursor c, CXCursor parent) {
        printf("Cursor kind: %u\n", clang_getCursorKind(c));
        return CXChildVisit_Recurse;
    };

    CXIndex index = clang_createIndex(0, 1);
    CXTranslationUnit unit = clang_parseTranslationUnit(
        index, "print-python-objects.c",
        (const char* const*) &argv[1], argc - 1,
        NULL, 0,
        CXTranslationUnit_None);

    CXCursor cursor = clang_getTranslationUnitCursor(unit);
    clang_visitChildrenWithBlock(cursor, block);

    clang_disposeTranslationUnit(unit);
    clang_disposeIndex(index);
}
