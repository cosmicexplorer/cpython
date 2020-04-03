#include "Python.h"
#include "frameobject.h" /* for PyFrame_ClearFreeList */
#include "pycore_context.h"
#include "pycore_initconfig.h"
#include "pycore_object.h"
#include "pycore_pyerrors.h"
#include "pycore_pymem.h"
#include "pycore_pystate.h"
#include "pydtrace.h"
#include "pytime.h" /* for _PyTime_GetMonotonicClock() */

#include "pdmp.h"

#include "stdbool.h"
#include "sys/mman.h"

/* PORTABLE DUMPING TIME!!!! */

static const char dump_magic[13] = {'D', 'U', 'M', 'P', 'E', 'D', 'C',
                                    'P', 'Y', 'T', 'H', 'O', 'N'};

/* This random fingerprint was generated by the shell command:

   shuf -i 0-255 -n 32 -r | awk '{printf "   0x%.02X,\n", $0}'

   In the final Emacs executable, this random fingerprint is replaced
   by a fingerprint of the temporary Emacs executable that was built
   along the way.  */

static const unsigned char fingerprint[32] = {
    0xDE, 0x86, 0xBB, 0x99, 0xFF, 0xF5, 0x46, 0x9A, 0x9E, 0x3F, 0x9F,
    0x5D, 0x9A, 0xDF, 0xF0, 0x91, 0xBD, 0xCD, 0xC1, 0xE8, 0x0C, 0x16,
    0x1E, 0xAF, 0xB8, 0x6C, 0xE2, 0x2B, 0xB1, 0x24, 0xCE, 0xB0,
};

static void pdmp_header_init(struct dump_header *header) {
    memcpy(header->magic, dump_magic, sizeof(dump_magic));
    memcpy(header->fingerprint, fingerprint, sizeof(fingerprint));
}

static bool pdmp_header_validate(struct dump_header *header) {
    return (memcmp(header->magic, dump_magic, sizeof(dump_magic)) == 0) &&
           (memcmp(header->fingerprint, fingerprint, sizeof(fingerprint)) == 0);
}

static PyObject *pdmp_dump_impl(struct dump_context *dump_context) {
    return NULL;
}

static PyObject *pdmp_dump_old(PyObject *Py_UNUSED(module), PyObject *obj,
                              PyObject *fd_obj)
{
    int fd;
    PyObject *dump_filename;
    struct dump_context dump_context;
    PyObject *result = Py_None;

    if (fd_obj == NULL || fd_obj == Py_None) {
        PyErr_SetString(PyExc_TypeError, "provided fd was NULL or None");
        goto error;
    }

    fd = PyObject_AsFileDescriptor(fd_obj);
    if (fd == -1) {
        PyErr_SetString(
            PyExc_TypeError,
            "provided fd did not point to an existing file descriptor");
        goto error;
    }

    dump_filename = PyObject_GetAttrString(fd_obj, "name");

    /* Initialize header. */
    pdmp_header_init(&dump_context.header);
    /* Initialize context. */
    dump_context.source_object = obj;
    dump_context.fd = fd;
    dump_context.dump_filename = dump_filename;

    result = pdmp_dump_impl(&dump_context);

    Py_XDECREF(dump_filename);

error:

    return result;
}

static int pdmp_map(pdmp *pdmp) {
    struct _Py_stat_struct status;
    if (_Py_fstat(pdmp->fd, &status)) {
        goto error;
    }
    if (status.st_size == 0) {
        PyErr_SetString(PyExc_ValueError, "pdmp mapped file had 0 bytes");
        goto error;
    }
    pdmp->mapped_memory_length = status.st_size;

    pdmp->mapped_memory_region =
        mmap(NULL, pdmp->mapped_memory_length, PROT_READ,
             MAP_FILE | MAP_PRIVATE, pdmp->fd, 0);
    if (pdmp->mapped_memory_region == MAP_FAILED) {
        PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError,
                                             pdmp->dump_filename);
        goto error;
    }

    pdmp->header = (struct dump_header *)pdmp->mapped_memory_region;
    pdmp->num_entries = (size_t *)(pdmp->header + sizeof(struct dump_header));
    pdmp->entries =
        (struct dump_reloc_entry *)(pdmp->num_entries + sizeof(size_t));
    pdmp->data = (void *)(pdmp->entries + ((*pdmp->num_entries) *
                                           sizeof(struct dump_reloc_entry *)));

    return 0;

error:
    pdmp->mapped_memory_length = -1;
    pdmp->mapped_memory_region = NULL;
    return -1;
}

static void pdmp_unmap(pdmp *pdmp) {
    munmap(pdmp->mapped_memory_region, pdmp->mapped_memory_length);
}

static PyObject *pdmp_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    pdmp *pdmp = type->tp_alloc(type, 0);
    if (pdmp == NULL) {
        return NULL;
    }

    pdmp->fd = -1;
    pdmp->dump_filename = Py_None;
    pdmp->mapped_memory_length = -1;
    pdmp->mapped_memory_region = NULL;
    pdmp->header = NULL;
    pdmp->num_entries = NULL;
    pdmp->entries = NULL;
    pdmp->data = NULL;

    return (PyObject *)pdmp;
}

static int pdmp_init(pdmp *self, PyObject *args, PyObject *kwds) {
    PyObject *file_handle = NULL;
    PyObject *dump_filename;
    int fd;

    if (!_PyArg_NoKeywords(Py_TYPE(self)->tp_name, kwds)) {
        return -1;
    }
    if (!PyArg_UnpackTuple(args, Py_TYPE(self)->tp_name, 1, 1, &file_handle)) {
        return -1;
    }

    fd = PyObject_AsFileDescriptor(file_handle);
    if (fd == -1) {
        PyErr_SetString(
            PyExc_TypeError,
            "provided fd did not point to an existing file descriptor");
        return -1;
    }

    dump_filename = PyObject_GetAttrString(file_handle, "name");

    self->fd = fd;
    self->dump_filename = dump_filename;

    return 0;
}

static bool pdmp_is_mapped(pdmp *self) {
    if (self->mapped_memory_length >= 0) {
        return true;
    }
    return false;
}

static void pdmp_dealloc(pdmp *pdmp) {
    _PyObject_GC_UNTRACK(pdmp);
    Py_XDECREF(pdmp->dump_filename);
    PyObject_GC_Del(pdmp);
}

static PyObject *pdmp_repr(pdmp *pdmp) {
    PyObject *result = NULL;
    int status = Py_ReprEnter((PyObject *)pdmp);

    if (status != 0) {
        if (status < 0) {
            return NULL;
        }
        result = PyUnicode_FromFormat("%s(...)", Py_TYPE(pdmp)->tp_name);
        goto done;
    }

    if (pdmp_is_mapped(pdmp)) {
        result = PyUnicode_FromFormat(
            "%s('%U', <mapped to %d bytes>)", Py_TYPE(pdmp)->tp_name,
            pdmp->dump_filename, pdmp->mapped_memory_length);
    } else {
        result =
            PyUnicode_FromFormat("%s('%U', <unmapped...>)",
                                 Py_TYPE(pdmp)->tp_name, pdmp->dump_filename);
    }

done:
    Py_ReprLeave((PyObject *)pdmp);
    return result;
}

PyDoc_STRVAR(pdmp_fileno_doc,
             "The file descriptor associated with the pdmp object.");

static PyObject *pdmp_fileno(pdmp *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromSsize_t(self->fd);
}

PyDoc_STRVAR(pdmp_filename_doc, "The name of the file associated with the pdmp object.");

static PyObject* pdmp_filename(pdmp* self, PyObject* Py_UNUSED(ignored)) {
    return self->dump_filename;
}

PyDoc_STRVAR(pdmp_enter_doc, "Mmap the associated file.");

static PyObject *pdmp_enter(pdmp *self, PyObject *Py_UNUSED(ignored)) {
    if (pdmp_is_mapped(self)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot re-map an already-mapped pdmp object");
        goto error;
    }

    if (pdmp_map(self)) {
        goto error;
    }

    /* FIXME: return the reconstructed object from `self->header.entry_point`, not the 'pdmp'
       instance itself! */
    return self;

error:
    return NULL;
}

PyDoc_STRVAR(pdmp_exit_doc, "Munmap the associated file.");

static PyObject *pdmp_exit(pdmp *self, PyObject *Py_UNUSED(ignored)) {
    pdmp_unmap(self);
    return Py_None;
}

PyDoc_STRVAR(pdmp_dump_doc, "Dump the object into the associated file.");

static PyObject *pdmp_dump(pdmp *self, PyObject *obj) {
    return NULL;
}

static PyMethodDef pdmp_methods[] = {
    {"fileno", (PyCFunction)pdmp_fileno, METH_NOARGS, pdmp_fileno_doc},
    {"filename", (PyCFunction)pdmp_filename, METH_NOARGS, pdmp_filename_doc},
    {"__enter__", (PyCFunction)pdmp_enter, METH_NOARGS, pdmp_enter_doc},
    {"__exit__", (PyCFunction)pdmp_exit, METH_VARARGS, pdmp_exit_doc},
    {"dump", (PyCFunction)pdmp_dump, METH_O, pdmp_dump_doc},
    {NULL, NULL} /* sentinel */
};

PyDoc_STRVAR(pdmp_doc,
             "pdmp(fd) -> new pdmp object from the given file handle\n\
\n\
Create a handle to an mmapped pdmp file.");

PyTypeObject Pdmp_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0) "pdmp", /* tp_name */
    sizeof(pdmp),                                  /* tp_basicsize */
    0,                                             /* tp_itemsize */
    /* methods */
    (destructor)pdmp_dealloc,    /* tp_dealloc */
    0,                           /* tp_vectorcall_offset */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    0,                           /* tp_as_async */
    (reprfunc)pdmp_repr,         /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    PyObject_HashNotImplemented, /* tp_hash */
    0,                           /* tp_call */
    0,                           /* tp_str */
    PyObject_GenericGetAttr,     /* tp_getattro */
    0,                           /* tp_setattro */
    0,                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
        Py_TPFLAGS_BASETYPE, /* tp_flags */
    pdmp_doc,                /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    pdmp_methods,            /* tp_methods */
    0,                       /* tp_members */
    0,                       /* tp_getset */
    0,                       /* tp_base */
    0,                       /* tp_dict */
    0,                       /* tp_descr_get */
    0,                       /* tp_descr_set */
    0,                       /* tp_dictoffset */
    (initproc)pdmp_init,     /* tp_init */
    PyType_GenericAlloc,     /* tp_alloc */
    pdmp_new,                /* tp_new */
    PyObject_GC_Del,         /* tp_free */
};
