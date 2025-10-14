/*
 * NO LICENSE IS AVAILABLE YET
 */

#ifndef STRING_OPS_INCLUDED
#define STRING_OPS_INCLUDED

typedef struct {
  PyObject_VAR_HEAD
  char to_match;
} SingleByteMatcher;

typedef enum {
  LEFT,
  RIGHT,
} SearchDirection;

typedef struct {
  PyObject_VAR_HEAD
  PyObject* string;
  Py_ssize_t cur_pos;
  SearchDirection direction;
} MatchProgress;

#endif  /* STRING_OPS_INCLUDED */
