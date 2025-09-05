/*
 * NO LICENSE IS AVAILABLE YET
 */

#include "Python.h"
#include "pycore_moduleobject.h" // _PyModule_GetState()

#include "string_ops.h"

static struct PyModuleDef stringopsmodule;

/* module state */
typedef struct {
  /* PyTypeObject *arg; */
} stringopsmodulestate;

static stringopsmodulestate *get_string_ops_module_state(PyObject *m) {
  stringopsmodulestate *state = (stringopsmodulestate *)_PyModule_GetState(m);
  assert(state);
  return state;
}

#define _string_ops_get_state_by_type(cls)                                     \
  get_string_ops_module_state(PyType_GetModuleByDef(cls, &stringopsmodule))

static PyMethodDef stringops_functions[] = {{NULL, NULL, 0, NULL}};

static int string_ops_exec(PyObject *m) {
  stringopsmodulestate *state;

  /* Create heap types */
  state = get_string_ops_module_state(m);
  /* state->arg = NULL; */

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
  /* Py_VISIT(state->arg); */
  return 0;
}

static int stringopsmodule_clear(PyObject *m) {
  stringopsmodulestate *state = get_string_ops_module_state(m);
  /* Py_CLEAR(state->arg); */
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
