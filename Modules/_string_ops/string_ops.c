/*
 * NO LICENSE IS AVAILABLE YET
 */

/* TODO: use PyBytesWriter for text replacement! */

#include "Python.h"
#include "pycore_long.h"          // _PyLong_GetZero()
#include "pycore_moduleobject.h"  // _PyModule_GetState()
#include "pycore_object.h"        // _PyObject_XSetRefDelayed()
#include "pycore_unicodeobject.h" // _PyUnicode_Copy()

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

/* clang-format off */

/*[clinic input]
module _string_ops
class _string_ops.STRINGOPS_ByteMatcher "SingleByteMatcher *" "get_string_ops_module_state_by_class(tp)->SingleByteMatcher"
class _string_ops.STRINGOPS_SearchDirection "SearchDirection *" "get_string_ops_module_state_by_class(tp)->SearchDirection"
class _string_ops.STRINGOPS_MatchProgress "MatchProgress *" "get_string_ops_module_state_by_class(tp)->MatchProgress"
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=7c9b7291970485e3]*/

/* clang-format on */

#include "clinic/string_ops.c.h"

static PyObject *byte_matcher_repr(PyObject *self) {
  SingleByteMatcher *obj = _SingleByteMatcher_CAST(self);
  return PyUnicode_FromFormat("_string_ops.ByteMatcher(%d)", obj->to_match);
}

static Py_hash_t byte_matcher_hash(PyObject *op) {
  SingleByteMatcher *self = _SingleByteMatcher_CAST(op);
  return self->to_match;
}

PyDoc_STRVAR(byte_matcher_doc, "Matcher for a single byte in a string.");

static PyObject *byte_matcher_richcompare(PyObject *lefto, PyObject *righto,
                                          int op) {
  PyTypeObject *tp = Py_TYPE(lefto);
  stringopsmodulestate *module_state = _string_ops_get_state_by_class(tp);
  SingleByteMatcher *left, *right;
  int cmp;

  if (!Py_IS_TYPE(righto, module_state->SingleByteMatcher)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  if (lefto == righto) {
    /* `is` relationship implies equality. */
    return PyBool_FromLong(op == Py_EQ);
  }
  left = (SingleByteMatcher *)lefto;
  right = (SingleByteMatcher *)righto;

  cmp = left->to_match == right->to_match;
  if (op == Py_NE) {
    cmp = !cmp;
  }
  return PyBool_FromLong(cmp);
}

#define BYTE_MATCHER_OFF(x) offsetof(SingleByteMatcher, x)
static PyMemberDef byte_matcher_members[] = {
    {"to_match", Py_T_BYTE, BYTE_MATCHER_OFF(to_match), Py_READONLY,
     "The numeric value of the byte this object matches against."},
    {NULL} /* Sentinel */
};

static PyMethodDef byte_matcher_methods[] = {
    {NULL, NULL}};

static PyType_Slot byte_matcher_slots[] = {
    {Py_tp_repr, byte_matcher_repr},
    {Py_tp_hash, byte_matcher_hash},
    {Py_tp_doc, byte_matcher_doc},
    {Py_tp_richcompare, byte_matcher_richcompare},
    {Py_tp_methods, byte_matcher_methods},
    {Py_tp_members, byte_matcher_members},
    {0, NULL},
};

static PyType_Spec byte_matcher_spec = {
    .name = "_string_ops.ByteMatcher",
    .basicsize = sizeof(SingleByteMatcher),
    .flags = (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE),
    .slots = byte_matcher_slots,
};

static PyObject *search_direction_repr(PyObject *self) {
  SearchDirection *obj = _SearchDirection_CAST(self);
  switch (obj->direction) {
  case LEFT:
    return PyUnicode_FromString("_string_ops.SearchDirection.LEFT()");
  case RIGHT:
    return PyUnicode_FromString("_string_ops.SearchDirection.RIGHT()");
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
  left = (SearchDirection *)lefto;
  right = (SearchDirection *)righto;

  cmp = left->direction == right->direction;
  if (op == Py_NE) {
    cmp = !cmp;
  }
  return PyBool_FromLong(cmp);
}

/* clang-format off */
static PyMethodDef search_direction_methods[] = {
    _STRING_OPS_STRINGOPS_SEARCHDIRECTION_LEFT_METHODDEF
    _STRING_OPS_STRINGOPS_SEARCHDIRECTION_RIGHT_METHODDEF
    {NULL, NULL},
};
/* clang-format on */

PyDoc_STRVAR(search_direction_doc,
             "Direction to begin a byte search in a string.");

static PyType_Slot search_direction_slots[] = {
    {Py_tp_repr, search_direction_repr},
    {Py_tp_hash, search_direction_hash},
    {Py_tp_doc, search_direction_doc},
    {Py_tp_richcompare, search_direction_richcompare},
    {Py_tp_methods, search_direction_methods},
    {0, NULL},
};

static PyType_Spec search_direction_spec = {
    .name = "_string_ops.SearchDirection",
    .basicsize = sizeof(SearchDirection),
    .flags = (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE),
    .slots = search_direction_slots,
};

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

PyDoc_STRVAR(match_progress_doc, "State necessary to track the progress of "
                                 "matching instances of a byte in a string.");

static PyType_Slot match_progress_slots[] = {
    {Py_tp_dealloc, match_progress_dealloc},
    /* {Py_tp_repr, match_progress_repr}, */
    {Py_tp_doc, match_progress_doc},
    /* {Py_tp_methods, match_progress_methods}, */
    {Py_tp_members, match_progress_members},
    {Py_tp_getset, match_progress_getset},
    {Py_tp_traverse, match_progress_traverse},
    {Py_tp_traverse, match_progress_clear},
    {0, NULL},
};

static PyType_Spec match_progress_spec = {
    .name = "_string_ops.MatchProgress",
    .basicsize = sizeof(MatchProgress),
    .flags = (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION |
              Py_TPFLAGS_HAVE_GC),
    .slots = match_progress_slots,
};

/* clang-format off */

/*[clinic input]
@classmethod
_string_ops.STRINGOPS_ByteMatcher.__new__

    byte: int

[clinic start generated code]*/

static PyObject *
_string_ops_STRINGOPS_ByteMatcher_impl(PyTypeObject *type, int byte)
/*[clinic end generated code: output=4767a234e617d813 input=32e29ff947165716]*/
{

  /* clang-format on */
  SingleByteMatcher *self = (SingleByteMatcher *)type->tp_alloc(type, 0);
  if (!self)
    return NULL;
  self->to_match = byte;
  PyObject_GC_Track(self);
  if (PyErr_Occurred()) {
    Py_DECREF(self);
    return NULL;
  }
  return (PyObject *)self;
  /* clang-format off */
}

/*[clinic input]
@classmethod
_string_ops.STRINGOPS_SearchDirection.LEFT

Begins matching at the end of the string and goes towards the beginning.
[clinic start generated code]*/

static PyObject *
_string_ops_STRINGOPS_SearchDirection_LEFT_impl(PyTypeObject *type)
/*[clinic end generated code: output=580bac40431fda8d input=153aca64f0c902fc]*/
{
  /* clang-format on */
  SearchDirection *self = (SearchDirection *)type->tp_alloc(type, 0);
  if (!self)
    return NULL;
  self->direction = LEFT;
  PyObject_GC_Track(self);
  if (PyErr_Occurred()) {
    Py_DECREF(self);
    return NULL;
  }
  return (PyObject *)self;
  /* clang-format off */
}

/*[clinic input]
@classmethod
_string_ops.STRINGOPS_SearchDirection.RIGHT

Begins matching at the start of the string and goes towards the end.
[clinic start generated code]*/

static PyObject *
_string_ops_STRINGOPS_SearchDirection_RIGHT_impl(PyTypeObject *type)
/*[clinic end generated code: output=3d278fb39e1757c0 input=0905c04f110e4b0b]*/
{
  /* clang-format on */
  SearchDirection *self = (SearchDirection *)type->tp_alloc(type, 0);
  if (!self)
    return NULL;
  self->direction = RIGHT;
  PyObject_GC_Track(self);
  if (PyErr_Occurred()) {
    Py_DECREF(self);
    return NULL;
  }
  return (PyObject *)self;
  /* clang-format off */
}

static PyMethodDef stringops_functions[] = {
    {NULL, NULL},
};

#define CREATE_TYPE(m, type, spec)                                             \
  do {                                                                         \
    type = (PyTypeObject *)PyType_FromModuleAndSpec(m, spec, NULL);            \
    if (type == NULL) {                                                        \
      goto error;                                                              \
    }                                                                          \
  } while (0)

static int string_ops_exec(PyObject *m) {
  stringopsmodulestate *state;

  /* Create heap types */
  state = get_string_ops_module_state(m);
  CREATE_TYPE(m, state->SingleByteMatcher, &byte_matcher_spec);
  CREATE_TYPE(m, state->SearchDirection, &search_direction_spec);
  CREATE_TYPE(m, state->MatchProgress, &match_progress_spec);

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
