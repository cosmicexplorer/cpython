/*
 * NO LICENSE IS AVAILABLE YET
 */

/* TODO: use PyBytesWriter for text replacement! */

#include "Python.h"
#include "pycore_long.h"          // _PyLong_GetZero()
#include "pycore_moduleobject.h"  // _PyModule_GetState()
#include "pycore_unicodeobject.h" // _PyUnicode_Copy

#include "string_ops.h"

static struct PyModuleDef stringopsmodule;

/* module state */
typedef struct {
  PyTypeObject *SingleByteMatcher;
  /* PyTypeObject *MatchProgress; */
} stringopsmodulestate;

static stringopsmodulestate *get_string_ops_module_state(PyObject *m) {
  stringopsmodulestate *state = (stringopsmodulestate *)_PyModule_GetState(m);
  assert(state);
  return state;
}

#define _string_ops_get_state_by_class(cls)                                    \
  get_string_ops_module_state(PyType_GetModule(cls))

#define _SingleByteMatcher_CAST(op) ((SingleByteMatcher *)(op))

/* clang-format off */
/*[clinic input]
module _string_ops
class _string_ops.STRINGOPS_ByteMatcher "SingleByteMatcher *" "get_string_ops_module_state_by_class(tp)->SingleByteMatcher"
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=4febd4ac65ce3de2]*/
/* clang-format on */

static PyObject *byte_matcher_repr(PyObject *self) {
  SingleByteMatcher *obj = _SingleByteMatcher_CAST(self);
  return PyUnicode_FromFormat("%c", obj->to_match);
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
    /* `is` relationship means equality. */
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

/* static PyObject *byte_matcher_byte(PyObject *op, void *Py_UNUSED(ignored)) {
 */
/*   SingleByteMatcher *self = _SingleByteMatcher_CAST(op); */
/*   return PyLong_FromLong(self->to_match); */
/* } */

/* static PyGetSetDef byte_matcher_getset[] = { */
/*     {"byte", byte_matcher_byte, NULL, */
/*       "The numeric value of the byte this object matches against."}, */
/*     {NULL}  /\* Sentinel *\/ */
/* }; */

#define BYTE_MATCHER_OFF(x) offsetof(SingleByteMatcher, x)
static PyMemberDef byte_matcher_members[] = {
    {"to_match", Py_T_INT, BYTE_MATCHER_OFF(to_match), Py_READONLY,
     "The numeric value of the byte this object matches against."},
    {NULL} /* Sentinel */
};

static PyMethodDef byte_matcher_methods[] = {{NULL, NULL}};

static PyType_Slot byte_matcher_slots[] = {
    {Py_tp_repr, byte_matcher_repr},
    {Py_tp_hash, byte_matcher_hash},
    {Py_tp_doc, (void *)byte_matcher_doc},
    {Py_tp_richcompare, byte_matcher_richcompare},
    {Py_tp_methods, byte_matcher_methods},
    {Py_tp_members, byte_matcher_members},
    /* {Py_tp_getset, byte_matcher_getset}, */
    {0, NULL},
};

static PyType_Spec byte_matcher_spec = {
    .name = "string_ops.ByteMatcher",
    .basicsize = sizeof(SingleByteMatcher),
    .flags = (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE |
              Py_TPFLAGS_DISALLOW_INSTANTIATION),
    .slots = byte_matcher_slots,
};

/* clang-format off */
/*[clinic input]
_string_ops.single_byte_matcher

    byte: int

[clinic start generated code]*/

static PyObject *
_string_ops_single_byte_matcher_impl(PyObject *module, int byte)
/*[clinic end generated code: output=51d0b2711fe82744 input=fbab89c8cb11dbce]*/
    /* clang-format on */

    static PyObject *_string_ops_single_byte_matcher_impl(PyObject *module,
                                                          char byte) {
  stringopsmodulestate *module_state = get_string_ops_module_state(module);
  SingleByteMatcher *self;
  self->to_match = byte;
  return (PyObject *)self;
}

#include "clinic/string_ops.c.h"

/* clang-format off */
static PyMethodDef stringops_functions[] = {
    _STRING_OPS_SINGLE_BYTE_MATCHER_METHODDEF
    {NULL, NULL},
};
/* clang-format on */

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
  return 0;
}

static int stringopsmodule_clear(PyObject *m) {
  stringopsmodulestate *state = get_string_ops_module_state(m);
  Py_CLEAR(state->SingleByteMatcher);
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
