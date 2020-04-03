/* Pdmp object interface */

#ifndef Py_PDMP_H
#define Py_PDMP_H
#ifdef __cplusplus
extern "C" {
#endif

static const char dump_magic[13];

static const unsigned char fingerprint[32];

struct dump_location {
    /* Index into the dump table entries. */
    size_t index;
};

struct dump_header {
    /* File type magic.  */
    char magic[sizeof dump_magic];

    unsigned char fingerprint[sizeof fingerprint];

    struct dump_location entry_point;
};

struct dump_reloc_entry {
    size_t base_offset;
    size_t extent;
};

typedef struct {
    PyObject_HEAD

    Py_ssize_t fd;
    PyObject* dump_filename;
    Py_ssize_t mapped_memory_length;
    void *mapped_memory_region;

    struct dump_header *header;
    size_t *num_entries;
    struct dump_reloc_entry *entries;
    void *data;
} pdmp;

struct dump_context {
    /* Header we'll write to the dump file when done.  */
    struct dump_header header;

    /* The object to dump. */
    PyObject *source_object;

    /* File descriptor for dump file; < 0 if closed.  */
    int fd;
    /* Name of dump file --- used for error reporting.  */
    PyObject *dump_filename;

    /* Queue of objects to dump.  */
    /* PyObject* dump_queue; */
};

PyAPI_DATA(PyTypeObject) Pdmp_Type;

#ifdef __cplusplus
}
#endif
#endif /* !Py_PDMP_H */
