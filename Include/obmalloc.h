#ifndef OBMALLOC_H
#define OBMALLOC_H
#ifdef __cplusplus
extern "C" {
#endif

#include "internal/pycore_atomic.h"

struct allocation_record {
    void *pointer;
    size_t nbytes;
};

struct all_allocations_report {
    struct allocation_record *all_allocations;
    size_t num_allocations;
};

struct all_allocations_report report_all_allocations(void);

void free_all_allocations_report(struct all_allocations_report*);

void *record_allocation(void *pointer, size_t nbytes);

#ifdef __cplusplus
}
#endif
#endif /* !OBMALLOC_H */
