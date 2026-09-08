#include "iree/hal/utils/deferred_work_queue.h"

typedef struct iree_hal_strela_semaphore_t {
  iree_async_semaphore_t async;
  iree_allocator_t host_allocator;
  iree_hal_deferred_work_queue_t* work_queue;
} iree_hal_strela_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_strela_semaphore_vtable;

static iree_hal_strela_semaphore_t *
iree_hal_strela_semaphore_cast(iree_hal_semaphore_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_semaphore_vtable);
  return (iree_hal_strela_semaphore_t *)base_value;
}

static iree_async_semaphore_t *
iree_hal_async_semaphore_cast(iree_hal_semaphore_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_semaphore_vtable);
  return (iree_async_semaphore_t *)base_value;
}

static iree_status_t
iree_hal_strela_semaphore_create(
  iree_async_proactor_t *proactor,
  iree_hal_queue_affinity_t queue_affinity,
  uint64_t initial_value,
  iree_hal_semaphore_flags_t flags,
  iree_hal_deferred_work_queue_t *work_queue,
  iree_allocator_t host_allocator,
  iree_hal_semaphore_t **out_semaphore
) {
  TRACE_FUNC;
  iree_status_t status = iree_ok_status();
  iree_hal_strela_semaphore_t *semaphore = NULL;
  iree_hal_semaphore_t *async_semaphore = NULL;
  iree_host_size_t frontier_offset = 0, total_size = 0;

  if (iree_status_is_ok(status)) {
    status = iree_async_semaphore_layout(
      sizeof *semaphore, 0, &frontier_offset, &total_size
    );
  }

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
      host_allocator, total_size, (void **)&semaphore
    );
  }

  if (iree_status_is_ok(status)) {
    iree_async_semaphore_initialize(
      (const iree_async_semaphore_vtable_t *)&iree_hal_strela_semaphore_vtable,
      proactor,
      initial_value,
      frontier_offset,
      0,
      &semaphore->async
    );
    semaphore->host_allocator = host_allocator;
    semaphore->work_queue = work_queue;
    async_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  }

  if (!iree_status_is_ok(status)) {
    if (async_semaphore) {
      iree_hal_semaphore_release(async_semaphore);
    }
  }

  *out_semaphore = async_semaphore;
  return status;
}

static void
iree_hal_strela_async_semaphore_destroy(iree_async_semaphore_t *base_semaphore) {
  TRACE_FUNC;
  iree_hal_strela_semaphore_t *semaphore = iree_hal_strela_semaphore_cast(
    iree_hal_semaphore_cast(base_semaphore)
  );
  iree_allocator_t host_allocator = semaphore->host_allocator;

  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);
}

static uint64_t
iree_hal_strela_async_semaphore_query(iree_async_semaphore_t *base_semaphore) {
  TRACE_FUNC;
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire
  );
  uint64_t value = 0;

  if (!iree_status_is_ok(failure)) {
    value = iree_hal_status_as_semaphore_failure(failure);
  } else {
    value = (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                       iree_memory_order_acquire);
  }

  return value;
}

static iree_status_t
iree_hal_strela_async_semaphore_signal(
  iree_async_semaphore_t *base_semaphore,
  uint64_t new_value,
  const iree_async_frontier_t *frontier
) {
  TRACE_FUNC;
  iree_hal_strela_semaphore_t *semaphore = iree_hal_strela_semaphore_cast(
    iree_hal_semaphore_cast(base_semaphore)
  );
  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    status = iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier
    );
  }

  if (iree_status_is_ok(status)) {
    iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
#if 0
  // TODO: write a class that implements the work_queue interface
    status = iree_hal_deferred_work_queue_issue(semaphore->work_queue);
#else
    (void)semaphore;
#endif
  }

  return status;
}

static void
iree_hal_strela_async_semaphore_on_fail(
  iree_async_semaphore_t *base_semaphore,
  iree_status_code_t status_code
) {
  TRACE_FUNC;
  iree_hal_strela_semaphore_t *semaphore = iree_hal_strela_semaphore_cast(
    iree_hal_semaphore_cast(base_semaphore)
  );

#if 0
  // TODO: write a class that implements the work_queue interface
  iree_status_ignore(iree_hal_deferred_work_queue_issue(semaphore->work_queue));
#else
  (void)semaphore;
#endif
}

static iree_status_t
iree_hal_strela_semaphore_wait(
  iree_hal_semaphore_t *base_semaphore,
  uint64_t value,
  iree_timeout_t timeout,
  iree_async_wait_flags_t flags
) {
  TRACE_FUNC;
  iree_async_semaphore_t *async_semaphore = iree_hal_async_semaphore_cast(base_semaphore);
  return iree_async_semaphore_multi_wait(
    IREE_ASYNC_WAIT_MODE_ALL,
    &async_semaphore,
    &value,
    /*count=*/1,
    timeout,
    flags,
    iree_allocator_system()
  );
}

static iree_status_t
iree_hal_strela_semaphore_import_timepoint(
  iree_hal_semaphore_t *base_semaphore,
  uint64_t value,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_external_timepoint_t external_timepoint
) {
  TRACE_FUNC;
  iree_hal_strela_semaphore_t *semaphore = iree_hal_strela_semaphore_cast(base_semaphore);

  (void)semaphore;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_semaphore_export_timepoint(
  iree_hal_semaphore_t *base_semaphore,
  uint64_t value,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_external_timepoint_type_t requested_type,
  iree_hal_external_timepoint_flags_t requested_flags,
  iree_hal_external_timepoint_t *out_external_timepoint
) {
  TRACE_FUNC;
  iree_hal_strela_semaphore_t *semaphore = iree_hal_strela_semaphore_cast(base_semaphore);

  (void)semaphore;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static const iree_hal_semaphore_vtable_t
iree_hal_strela_semaphore_vtable = {
  .async = {
    .destroy = iree_hal_strela_async_semaphore_destroy,
    .query   = iree_hal_strela_async_semaphore_query,
    .signal  = iree_hal_strela_async_semaphore_signal,
    .on_fail = iree_hal_strela_async_semaphore_on_fail,
  },
  .wait             = iree_hal_strela_semaphore_wait,
  .import_timepoint = iree_hal_strela_semaphore_import_timepoint,
  .export_timepoint = iree_hal_strela_semaphore_export_timepoint,
};
