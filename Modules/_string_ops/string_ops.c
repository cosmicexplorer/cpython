/*
 * NO LICENSE IS AVAILABLE YET
 */

/* TODO: use PyBytesWriter for text replacement! */

#include "Python.h"
#include "pycore_long.h"          // _PyLong_GetZero()
#include "pycore_moduleobject.h"  // _PyModule_GetState()
#include "pycore_object.h"        // _PyObject_XSetRefDelayed()
#include "pycore_unicodeobject.h" // _PyUnicode_Copy()

#if defined(Py_BUILD_CORE) && !defined(Py_BUILD_CORE_MODULE)
#include "pycore_gc.h"      // PyGC_Head
#include "pycore_runtime.h" // _Py_ID()
#endif
#include "pycore_modsupport.h" // _PyArg_UnpackKeywords()

#include "string_ops.h"

static struct PyModuleDef stringopsmodule;

/* module state */
typedef struct {
  PyTypeObject *SingleByteMatcher;
  PyTypeObject *SearchDirection;
  PyTypeObject *MatchProgress;
} stringopsmodulestate;

static stringopsmodulestate *get_string_ops_module_state(PyObject *m) {
  stringopsmodulestate *state = (stringopsmodulestate *)_PyModule_GetState(m);
  assert(state);
  return state;
}

#define _string_ops_get_state_by_class(cls)                                    \
  get_string_ops_module_state(PyType_GetModule(cls))

#define _SingleByteMatcher_CAST(op) ((SingleByteMatcher *)(op))
#define _SearchDirection_CAST(op) ((SearchDirection *)(op))
#define _MatchProgress_CAST(op) ((MatchProgress *)(op))

static PyObject *_string_ops_STRINGOPS_ByteMatcher(PyTypeObject *subtype,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  assert(PyType_Check(subtype));

  assert(kwargs == NULL || PyDict_Check(kwargs));
  if (kwargs != NULL && PyDict_GET_SIZE(kwargs)) {
    const char *msg = "ByteMatcher() does not accept kwargs";
    PyErr_Format(PyExc_TypeError, msg);
    return NULL;
  }

  assert(args == NULL || PyTuple_Check(args));
  Py_ssize_t len = (args != NULL) ? PyTuple_GET_SIZE(args) : 0;
  if (len > 1) {
    const char *msg =
        "ByteMatcher() takes at most 1 positional argument (%zd given)";
    PyErr_Format(PyExc_TypeError, msg, len);
    return NULL;
  } else if (!len) {
    const char *msg =
        "ByteMatcher() requires at least 1 positional argument (%zd given)";
    PyErr_Format(PyExc_TypeError, msg, len);
    return NULL;
  }

  assert(len == 1);
  SingleByteMatcher *self =
      _SingleByteMatcher_CAST(subtype->tp_alloc(subtype, 0));
  if (!self) {
    return NULL;
  }
  int arg = PyLong_AsInt(PyTuple_GET_ITEM(args, 0));
  if (arg == -1 && PyErr_Occurred()) {
    return NULL;
  }
  if (arg < 0) {
    const char *msg = "ByteMatcher() only works on positive bytes (%d given)";
    PyErr_Format(PyExc_ValueError, msg, arg);
    return NULL;
  }
  if (arg > 255) {
    const char *msg =
        "ByteMatcher() only works on individual bytes < 256 (%zd given)";
    PyErr_Format(PyExc_ValueError, msg, arg);
    return NULL;
  }
  self->to_match = arg;

  return (PyObject *)self;
}

static PyObject *byte_matcher_repr(PyObject *self) {
  SingleByteMatcher *obj = _SingleByteMatcher_CAST(self);
  return PyUnicode_FromFormat("ByteMatcher(%d)", obj->to_match);
}

static Py_hash_t byte_matcher_hash(PyObject *op) {
  SingleByteMatcher *self = _SingleByteMatcher_CAST(op);
  return self->to_match;
}

static PyObject *byte_matcher_richcompare(PyObject *lefto, PyObject *righto,
                                          int op) {
  PyTypeObject *tp = Py_TYPE(lefto);
  stringopsmodulestate *module_state = _string_ops_get_state_by_class(tp);
  SingleByteMatcher *left, *right;

  if (!Py_IS_TYPE(righto, module_state->SingleByteMatcher)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  if (lefto == righto) {
    /* `is` relationship implies equality. */
    return PyBool_FromLong(op == Py_EQ);
  }
  left = _SingleByteMatcher_CAST(lefto);
  right = _SingleByteMatcher_CAST(righto);

  switch (op) {
  case Py_EQ:
    return PyBool_FromLong(left->to_match == right->to_match);
  case Py_NE:
    return PyBool_FromLong(left->to_match != right->to_match);
  case Py_LT:
    return PyBool_FromLong(left->to_match < right->to_match);
  case Py_GT:
    return PyBool_FromLong(left->to_match > right->to_match);
  case Py_LE:
    return PyBool_FromLong(left->to_match <= right->to_match);
  case Py_GE:
    return PyBool_FromLong(left->to_match >= right->to_match);
  default:
    abort();
  }
}

#define BYTE_MATCHER_OFF(x) offsetof(SingleByteMatcher, x)
static PyMemberDef byte_matcher_members[] = {
    {"to_match", Py_T_BYTE, BYTE_MATCHER_OFF(to_match), Py_READONLY,
     "The numeric value of the byte this object matches against."},
    {NULL} /* Sentinel */
};

/* clang-format off */
static PyTypeObject ByteMatcherType = {
  PyObject_HEAD_INIT(NULL)
  .tp_basicsize = sizeof(SingleByteMatcher),
  .tp_new = _string_ops_STRINGOPS_ByteMatcher,
  .tp_name = "_string_ops.ByteMatcher",
  .tp_doc = PyDoc_STR("ByteMatcher(byte, /)\n"
                      "--\n\n"
                      "Matcher for a single byte in a string."),
  .tp_flags = Py_TPFLAGS_IMMUTABLETYPE,
  .tp_repr = byte_matcher_repr,
  .tp_hash = byte_matcher_hash,
  .tp_richcompare = byte_matcher_richcompare,
  .tp_members = byte_matcher_members,
};
/* clang-format on */

static PyObject *search_direction_base_new(PyTypeObject *subtype,
                                           PyObject *args, PyObject *kwargs) {
  assert(PyType_Check(subtype));

  assert(kwargs == NULL || PyDict_Check(kwargs));
  if (kwargs != NULL && PyDict_GET_SIZE(kwargs)) {
    const char *msg = "SearchDirection() does not accept kwargs";
    PyErr_Format(PyExc_TypeError, msg);
    return NULL;
  }

  assert(args == NULL || PyTuple_Check(args));
  Py_ssize_t len = (args != NULL) ? PyTuple_GET_SIZE(args) : 0;
  if (len > 1) {
    const char *msg =
        "_SearchDirection() takes at most 1 positional argument (%zd given)";
    PyErr_Format(PyExc_TypeError, msg, len);
    return NULL;
  } else if (!len) {
    const char *msg = "_SearchDirection() requires at least 1 positional "
                      "argument (%zd given)";
    PyErr_Format(PyExc_TypeError, msg, len);
    return NULL;
  }

  assert(len == 1);
  SearchDirection *self = _SearchDirection_CAST(subtype->tp_alloc(subtype, 0));
  if (!self) {
    return NULL;
  }
  int arg = PyLong_AsInt(PyTuple_GET_ITEM(args, 0));
  if (arg == -1 && PyErr_Occurred()) {
    return NULL;
  }
  switch (arg) {
  case LEFT:
    self->direction = LEFT;
    break;
  case RIGHT:
    self->direction = RIGHT;
    break;
  default:
    const char *msg = "_SearchDirection() requires either LEFT(%zd) or "
                      "RIGHT(%zd) as an argument (%zd given)";
    PyErr_Format(PyExc_KeyError, msg, LEFT, RIGHT, arg);
    return NULL;
  }
  self->direction = arg;

  return (PyObject *)self;
}

static PyObject *search_direction_repr(PyObject *self) {
  SearchDirection *obj = _SearchDirection_CAST(self);
  switch (obj->direction) {
  case LEFT:
    return PyUnicode_FromString("SearchDirection.LEFT");
  case RIGHT:
    return PyUnicode_FromString("SearchDirection.RIGHT");
  default:
    abort();
  }
}

static Py_hash_t search_direction_hash(PyObject *op) {
  SearchDirection *self = _SearchDirection_CAST(op);
  return self->direction;
}

static PyObject *search_direction_richcompare(PyObject *lefto, PyObject *righto,
                                              int op) {
  PyTypeObject *tp = Py_TYPE(lefto);
  stringopsmodulestate *module_state = _string_ops_get_state_by_class(tp);
  SearchDirection *left, *right;
  int cmp;

  if (!Py_IS_TYPE(righto, module_state->SearchDirection)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  if (lefto == righto) {
    /* `is` relationship implies equality. */
    return PyBool_FromLong(op == Py_EQ);
  }
  left = _SearchDirection_CAST(lefto);
  right = _SearchDirection_CAST(righto);

  cmp = left->direction == right->direction;
  if (op == Py_NE) {
    cmp = !cmp;
  }
  return PyBool_FromLong(cmp);
}

/* clang-format off */
static PyTypeObject SearchDirectionType = {
  PyObject_HEAD_INIT(NULL)
  .tp_basicsize = sizeof(SearchDirection),
  .tp_new = search_direction_base_new,
  .tp_name = "_string_ops._SearchDirection",
  .tp_doc = PyDoc_STR("SearchDirection\n"
             "--\n"
             "\n"
             "Direction to begin a byte search in a string."),
  .tp_flags = Py_TPFLAGS_BASETYPE,
  .tp_repr = search_direction_repr,
  .tp_hash = search_direction_hash,
  .tp_richcompare = search_direction_richcompare,
};
/* clang-format on */

static int match_progress_traverse(PyObject *op, visitproc visit, void *arg) {
  MatchProgress *self = _MatchProgress_CAST(op);
  Py_VISIT(Py_TYPE(self));
  Py_VISIT(self->string);
  Py_VISIT(self->direction);
  Py_VISIT(self->matcher);
  return 0;
}

static int match_progress_clear(PyObject *op) {
  MatchProgress *self = _MatchProgress_CAST(op);
  Py_CLEAR(self->string);
  Py_CLEAR(self->direction);
  Py_CLEAR(self->matcher);
  return 0;
}

static void match_progress_dealloc(PyObject *self) {
  PyTypeObject *tp = Py_TYPE(self);
  PyObject_GC_UnTrack(self);
  (void)match_progress_clear(self);
  tp->tp_free(self);
  Py_DECREF(tp);
}

#define MATCH_OFF(x) offsetof(MatchProgress, x)
static PyMemberDef match_progress_members[] = {
    {"string", Py_T_STRING, MATCH_OFF(string), Py_READONLY,
     "The target string data on which this byte matcher is being applied."},
    {"cur_pos", Py_T_INT, MATCH_OFF(cur_pos), Py_READONLY,
     "The position in the string at which the byte last matched. "
     "This will be -1 if it has not been attempted to be matched yet, "
     "or 1 past the end of the string if all matches have been found. "
     "This will be reversed for LEFT direction searches."},
    {NULL} /* Sentinel. */
};

static PyObject *match_progress_get_direction(PyObject *op,
                                              void *Py_UNUSED(ignored)) {
  MatchProgress *self = _MatchProgress_CAST(op);
  return _Py_XNewRef((PyObject *)self->direction);
}

static int match_progress_set_direction(PyObject *op, PyObject *obj,
                                        void *Py_UNUSED(ignored)) {
  PyTypeObject *tp = Py_TYPE(op);
  stringopsmodulestate *module_state = _string_ops_get_state_by_class(tp);
  MatchProgress *self = _MatchProgress_CAST(op);
  if (obj == NULL || !Py_IS_TYPE(obj, module_state->SearchDirection)) {
    PyErr_SetString(
        PyExc_TypeError,
        "direction must be set to a _string_ops.SearchDirection object");
    return -1;
  }
  Py_BEGIN_CRITICAL_SECTION(self);
  _PyObject_XSetRefDelayed(&self->direction, Py_NewRef(obj));
  Py_END_CRITICAL_SECTION();
  return 0;
}

static PyObject *match_progress_get_matcher(PyObject *op,
                                            void *Py_UNUSED(ignored)) {
  MatchProgress *self = _MatchProgress_CAST(op);
  return _Py_XNewRef((PyObject *)self->matcher);
}

static int match_progress_set_matcher(PyObject *op, PyObject *obj,
                                      void *Py_UNUSED(ignored)) {
  PyTypeObject *tp = Py_TYPE(op);
  stringopsmodulestate *module_state = _string_ops_get_state_by_class(tp);
  MatchProgress *self = _MatchProgress_CAST(op);
  if (obj == NULL || !Py_IS_TYPE(obj, module_state->SingleByteMatcher)) {
    PyErr_SetString(
        PyExc_TypeError,
        "matcher must be set to a _string_ops.SingleByteMatcher object");
    return -1;
  }
  Py_BEGIN_CRITICAL_SECTION(self);
  _PyObject_XSetRefDelayed(&self->direction, Py_NewRef(obj));
  Py_END_CRITICAL_SECTION();
  return 0;
}

static PyGetSetDef match_progress_getset[] = {
    {"direction", match_progress_get_direction, match_progress_set_direction,
     PyDoc_STR("The direction to perform further search operations in."), NULL},
    {"matcher", match_progress_get_matcher, match_progress_set_matcher,
     PyDoc_STR("The byte to perform matching against."), NULL},
    {NULL} /* Sentinel */
};

/* clang-format off */
static PyTypeObject MatchProgressType = {
  PyObject_HEAD_INIT(NULL)
  .tp_basicsize = sizeof(MatchProgress),
  .tp_name = "_string_ops.MatchProgress",
  .tp_doc = PyDoc_STR("MatchProgress()\n"
                      "--\n\n"
                      "State necessary to track the progress of "
                      "matching instances of a byte in a string."),
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
  .tp_members = match_progress_members,
  .tp_getset = match_progress_getset,
  .tp_traverse = match_progress_traverse,
  .tp_clear = match_progress_clear,
  .tp_dealloc = match_progress_dealloc,
};
/* clang-format on */

static PyMethodDef stringops_functions[] = {
    {NULL, NULL},
};

#define ADD_TYPE(m, type)                                                      \
  do {                                                                         \
    if (PyModule_AddType(m, type) < 0) {                                       \
      goto error;                                                              \
    }                                                                          \
  } while (0)

static int string_ops_exec(PyObject *m) {
  stringopsmodulestate *state;

  state = get_string_ops_module_state(m);

  state->SingleByteMatcher = &ByteMatcherType;
  state->SearchDirection = &SearchDirectionType;
  state->MatchProgress = &MatchProgressType;

  ADD_TYPE(m, state->SingleByteMatcher);
  ADD_TYPE(m, state->SearchDirection);

  if (PyModule_AddIntConstant(m, "MAGIC", 777) < 0) {
    goto error;
  }

  return 0;

error:
  return -1;
}

static PyModuleDef_Slot string_ops_slots[] = {
    {Py_mod_exec, string_ops_exec},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
    {0, NULL},
};

static int stringopsmodule_traverse(PyObject *m, visitproc visit, void *arg) {
  stringopsmodulestate *state = get_string_ops_module_state(m);
  Py_VISIT(state->SingleByteMatcher);
  Py_VISIT(state->SearchDirection);
  Py_VISIT(state->MatchProgress);
  return 0;
}

static int stringopsmodule_clear(PyObject *m) {
  stringopsmodulestate *state = get_string_ops_module_state(m);
  Py_CLEAR(state->SingleByteMatcher);
  Py_CLEAR(state->SearchDirection);
  Py_CLEAR(state->MatchProgress);
  return 0;
}

static void stringopsmodule_free(void *m) {
  stringopsmodule_clear((PyObject *)m);
}

static struct PyModuleDef stringopsmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_string_ops",
    .m_size = sizeof(stringopsmodulestate),
    .m_methods = stringops_functions,
    .m_slots = string_ops_slots,
    .m_traverse = stringopsmodule_traverse,
    .m_clear = stringopsmodule_clear,
    .m_free = stringopsmodule_free,
};

PyMODINIT_FUNC PyInit__string_ops(void) {
  return PyModuleDef_Init(&stringopsmodule);
}
