/*[clinic input]
preserve
[clinic start generated code]*/

#if defined(Py_BUILD_CORE) && !defined(Py_BUILD_CORE_MODULE)
#  include "pycore_gc.h"          // PyGC_Head
#  include "pycore_runtime.h"     // _Py_ID()
#endif
#include "pycore_modsupport.h"    // _PyArg_UnpackKeywords()

PyDoc_STRVAR(_string_ops_single_byte_matcher__doc__,
"single_byte_matcher($module, /, byte)\n"
"--\n"
"\n");

#define _STRING_OPS_SINGLE_BYTE_MATCHER_METHODDEF    \
    {"single_byte_matcher", _PyCFunction_CAST(_string_ops_single_byte_matcher), METH_FASTCALL|METH_KEYWORDS, _string_ops_single_byte_matcher__doc__},

static PyObject *
_string_ops_single_byte_matcher_impl(PyObject *module, int byte);

static PyObject *
_string_ops_single_byte_matcher(PyObject *module, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *return_value = NULL;
    #if defined(Py_BUILD_CORE) && !defined(Py_BUILD_CORE_MODULE)

    #define NUM_KEYWORDS 1
    static struct {
        PyGC_Head _this_is_not_used;
        PyObject_VAR_HEAD
        Py_hash_t ob_hash;
        PyObject *ob_item[NUM_KEYWORDS];
    } _kwtuple = {
        .ob_base = PyVarObject_HEAD_INIT(&PyTuple_Type, NUM_KEYWORDS)
        .ob_hash = -1,
        .ob_item = { &_Py_ID(bytes), },
    };
    #undef NUM_KEYWORDS
    #define KWTUPLE (&_kwtuple.ob_base.ob_base)

    #else  // !Py_BUILD_CORE
    #  define KWTUPLE NULL
    #endif  // !Py_BUILD_CORE

    static const char * const _keywords[] = {"byte", NULL};
    static _PyArg_Parser _parser = {
        .keywords = _keywords,
        .fname = "single_byte_matcher",
        .kwtuple = KWTUPLE,
    };
    #undef KWTUPLE
    PyObject *argsbuf[1];
    int byte;

    args = _PyArg_UnpackKeywords(args, nargs, NULL, kwnames, &_parser,
            /*minpos*/ 1, /*maxpos*/ 1, /*minkw*/ 0, /*varpos*/ 0, argsbuf);
    if (!args) {
        goto exit;
    }
    byte = PyLong_AsInt(args[0]);
    if (byte == -1 && PyErr_Occurred()) {
        goto exit;
    }
    return_value = _string_ops_single_byte_matcher_impl(module, byte);

exit:
    return return_value;
}
/*[clinic end generated code: output=48368341f5e7aa35 input=a9049054013a1b77]*/
