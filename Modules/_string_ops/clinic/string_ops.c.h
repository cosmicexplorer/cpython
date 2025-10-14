/*[clinic input]
preserve
[clinic start generated code]*/

#if defined(Py_BUILD_CORE) && !defined(Py_BUILD_CORE_MODULE)
#  include "pycore_gc.h"          // PyGC_Head
#  include "pycore_runtime.h"     // _Py_ID()
#endif
#include "pycore_modsupport.h"    // _PyArg_UnpackKeywords()

static PyObject *
_string_ops_STRINGOPS_ByteMatcher_impl(PyTypeObject *type, int byte);

static PyObject *
_string_ops_STRINGOPS_ByteMatcher(PyTypeObject *type, PyObject *args, PyObject *kwargs)
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
        .fname = "STRINGOPS_ByteMatcher",
        .kwtuple = KWTUPLE,
    };
    #undef KWTUPLE
    PyObject *argsbuf[1];
    PyObject * const *fastargs;
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    int byte;

    fastargs = _PyArg_UnpackKeywords(_PyTuple_CAST(args)->ob_item, nargs, kwargs, NULL, &_parser,
            /*minpos*/ 1, /*maxpos*/ 1, /*minkw*/ 0, /*varpos*/ 0, argsbuf);
    if (!fastargs) {
        goto exit;
    }
    byte = PyLong_AsInt(fastargs[0]);
    if (byte == -1 && PyErr_Occurred()) {
        goto exit;
    }
    return_value = _string_ops_STRINGOPS_ByteMatcher_impl(type, byte);

exit:
    return return_value;
}

PyDoc_STRVAR(_string_ops_STRINGOPS_SearchDirection_LEFT__doc__,
"LEFT($type, /)\n"
"--\n"
"\n"
"Begins matching at the end of the string and goes towards the beginning.");

#define _STRING_OPS_STRINGOPS_SEARCHDIRECTION_LEFT_METHODDEF    \
    {"LEFT", (PyCFunction)_string_ops_STRINGOPS_SearchDirection_LEFT, METH_NOARGS|METH_CLASS, _string_ops_STRINGOPS_SearchDirection_LEFT__doc__},

static PyObject *
_string_ops_STRINGOPS_SearchDirection_LEFT_impl(PyTypeObject *type);

static PyObject *
_string_ops_STRINGOPS_SearchDirection_LEFT(PyObject *type, PyObject *Py_UNUSED(ignored))
{
    return _string_ops_STRINGOPS_SearchDirection_LEFT_impl((PyTypeObject *)type);
}

PyDoc_STRVAR(_string_ops_STRINGOPS_SearchDirection_RIGHT__doc__,
"RIGHT($type, /)\n"
"--\n"
"\n"
"Begins matching at the start of the string and goes towards the end.");

#define _STRING_OPS_STRINGOPS_SEARCHDIRECTION_RIGHT_METHODDEF    \
    {"RIGHT", (PyCFunction)_string_ops_STRINGOPS_SearchDirection_RIGHT, METH_NOARGS|METH_CLASS, _string_ops_STRINGOPS_SearchDirection_RIGHT__doc__},

static PyObject *
_string_ops_STRINGOPS_SearchDirection_RIGHT_impl(PyTypeObject *type);

static PyObject *
_string_ops_STRINGOPS_SearchDirection_RIGHT(PyObject *type, PyObject *Py_UNUSED(ignored))
{
    return _string_ops_STRINGOPS_SearchDirection_RIGHT_impl((PyTypeObject *)type);
}
/*[clinic end generated code: output=3e66e2d3db00349f input=a9049054013a1b77]*/
