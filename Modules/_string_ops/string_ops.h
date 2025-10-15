/*
 * NO LICENSE IS AVAILABLE YET
 */

#ifndef STRING_OPS_INCLUDED
#define STRING_OPS_INCLUDED

typedef struct {
  PyObject_HEAD
  unsigned char to_match;
} ByteMatcher;

typedef enum {
  LEFT = 0,
  RIGHT = 1,
} SearchDirectionData;

typedef struct {
  PyObject_HEAD
  SearchDirectionData direction;
} SearchDirection;

typedef struct {
  PyObject_HEAD
  PyObject* data_block;
  Py_ssize_t cur_pos;
  PyObject* direction;
  PyObject* matcher;
} MatchProgress;

#endif  /* STRING_OPS_INCLUDED */
