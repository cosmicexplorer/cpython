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
} SearchDirectionData;

typedef struct {
  PyObject_VAR_HEAD
  SearchDirectionData direction;
} SearchDirection;

typedef struct {
  PyObject_VAR_HEAD
  PyObject* string;
  Py_ssize_t cur_pos;
  PyObject* direction;
  PyObject* matcher;
} MatchProgress;

#endif  /* STRING_OPS_INCLUDED */
