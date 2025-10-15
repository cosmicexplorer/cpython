/*
 * NO LICENSE IS AVAILABLE YET
 */

#ifndef STRING_OPS_SIMD_INCLUDED
#define STRING_OPS_SIMD_INCLUDED

typedef enum {
  UNSTABLE = 0,
} MatcherVersion;

typedef struct {
  MatcherVersion version;
  unsigned char to_match;
} CompiledMatcher;

typedef enum {
  LEFT_TO_RIGHT = 0,
  RIGHT_TO_LEFT = 1,
} SearchDirection;

typedef struct {
  const char *data;
  const size_t extent;
} ReadBuffer;

typedef struct {
  ReadBuffer buf;
  SearchDirection direction;
} SingleMatchRequest;

/* clang-format off */
/* Maintaining correct bookkeeping for iterative matching over the same input is
   extremely difficult:
   https://codeberg.org/cosmicexplorer/deep-link/src/commit/1ea3eba5d599d8c48ea56816b7103b12ee49d505/trait-utils/src/bounds.rs#L29

   We don't have rust iterators, but we *do* have something more important:
   (1) Explicitly-controlled memory buffers which can be used to achieve pipelining:
   https://docs.python.org/3/library/io.html#io.BufferedRWPair
   (2) An explicit interface for stateful (stream) encoding:
   https://docs.python.org/3/library/codecs.html#stream-encoding-and-decoding
   (3) Some conventions for providing "translations" that avoid calling back into Python at all:
   https://docs.python.org/3/library/stdtypes.html#bytes.translate

   Along with buffering and streaming, there are also other encoding processes that occur extremely
   frequently in performance-sensitive tooling. In particular, URL quoting as well as parsing more
   generally has become a significant performance hotspot for pip resolves[^1].

   Python now has access to shared state not just in the same process, but also across processes, in
   forms that present a contiguous array of bytes as an abstraction:
   https://docs.python.org/3/library/multiprocessing.html#shared-ctypes-objects
   https://docs.python.org/3/library/mmap.html

   The first thing users see when looking up the json module is warning about denial of service:
   https://docs.python.org/3/library/json.html
   Indeed, if we look to the expat project, we find that exposing text and bytes as a sort of
   coroutine-like idea with callbacks is often a way to improve parsing robustness and performance:
   https://www.xml.com/pub/1999/09/expat/index.html

   Finally, while we have the translation table abstraction available for mapping individual bytes
   to others, each codec implementation still has to handle variable-length in its own way, usually
   through ''.join().

   So, we propose two goals:
   - Describe a coroutine interface for fully-general end-user usage of fast string search.
   - Describe an informal interface for mapping bytes or strings to a growable output buffer.
     - This would take advantage of mostly staying within C loops, as noted in the current
       urllib.parse.quoter() impl:
       https://github.com/python/cpython/blob/0bcb1c25f7ba254bea9b744c7c5423cfebade3b3/Lib/urllib/parse.py#L883-L884

   [^1]: source: personal testing
*/
/* clang-format on */

typedef enum {
  BEFORE_START = 0,
  AFTER_END = 1,
  MATCHED_INTERNAL_POSITION = 2,
} SingleMatchResultKind;

typedef struct {
  SingleMatchResultKind kind;
  size_t length_between_matches;
} SingleMatchResult;

/* Iterate the coroutine! */
SingleMatchResult single_string_match(const SingleMatchRequest &req,
                                      CompiledMatcher &matcher);

#endif /* STRING_OPS_SIMD_INCLUDED */
