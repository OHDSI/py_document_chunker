import pytest

from py_document_chunker import CodeSplitter

# A comprehensive Java code sample to test various top-level declarations.
SAMPLE_JAVA_CODE = """
package com.example.geometry;

/**
 * A utility class for geometry calculations.
 * This class is final.
 */
public final class GeometryUtils {

    public static final double PI = 3.14159;

    // A simple constructor
    public GeometryUtils() {
        System.out.println("Utils created.");
    }

    /**
     * Calculates the area of a circle.
     * @param radius The radius of the circle.
     * @return The area of the circle.
     */
    public static double calculateCircleArea(double radius) {
        return PI * radius * radius;
    }
}

/**
 * Interface for shapes that can be drawn.
 */
interface Drawable {
    void draw();
    String getShapeName();
}

// A record for representing a 2D point
public record Point(int x, int y) {}

// An enum for basic colors
enum Color {
    RED, GREEN, BLUE
}

@interface MyAnnotation {
    String value();
}
"""


def test_java_code_splitter_respects_top_level_boundaries():
    """
    Tests that the CodeSplitter correctly splits Java code based on high-level
    syntactic boundaries like classes, interfaces, enums, etc.
    """
    # This test requires the 'code' extras to be installed.
    try:
        from tree_sitter_language_pack import get_language

        get_language("java")
    except (ImportError, Exception):
        pytest.skip("Java tree-sitter grammar not available. Skipping test.")

    # The chunk_size is deliberately set high to ensure that splitting occurs
    # due to syntactic boundaries, not because a chunk is too large.
    splitter = CodeSplitter(language="java", chunk_size=1024, chunk_overlap=0)
    chunks = splitter.split_text(SAMPLE_JAVA_CODE)

    # The new logic correctly finds the 5 top-level chunkable declarations.
    # The GeometryUtils class is treated as a single chunk because it is smaller
    # than the chunk_size.
    assert len(chunks) == 5

    # Verify the content of each chunk to ensure they match the declarations.
    # We use .strip() to be robust against minor whitespace differences.
    chunk_contents = [c.content.strip() for c in chunks]

    assert chunk_contents[0].startswith("public final class GeometryUtils")
    assert chunk_contents[0].endswith("}")

    assert chunk_contents[1].startswith("interface Drawable")
    assert chunk_contents[1].endswith("}")

    assert chunk_contents[2] == "public record Point(int x, int y) {}"

    assert chunk_contents[3].startswith("enum Color")
    assert chunk_contents[3].endswith("}")

    assert chunk_contents[4].startswith("@interface MyAnnotation")
    assert chunk_contents[4].endswith("}")
