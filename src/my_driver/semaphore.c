typedef struct strela_semaphore_t {
  iree_async_semaphore_t async;
  iree_allocator_t host_allocator;
  iree_atomic_int64_t payload_value;
} strela_semaphore_t;

static iree_status_t
strela_semaphore_wait(iree_hal_semaphore_t* semaphore, uint64_t value, iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_semaphore_import_timepoint(iree_hal_semaphore_t* semaphore, uint64_t value, iree_hal_queue_affinity_t queue_affinity, iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_semaphore_export_timepoint(iree_hal_semaphore_t* semaphore, uint64_t value, iree_hal_queue_affinity_t queue_affinity, iree_hal_external_timepoint_type_t requested_type, iree_hal_external_timepoint_flags_t requested_flags, iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static void
strela_semaphore_destroy(iree_async_semaphore_t* semaphore) {
  printf("%s\n", __func__);
}

static uint64_t
strela_semaphore_query(iree_async_semaphore_t *base_semaphore) {
  strela_semaphore_t* semaphore = (strela_semaphore_t*)base_semaphore;
  return iree_atomic_load(&semaphore->payload_value, iree_memory_order_acquire);
}

static iree_status_t
strela_semaphore_signal(
  iree_async_semaphore_t *base_semaphore,
  uint64_t value,
  const iree_async_frontier_t* frontier
) {
  printf("%s: signaling to %llu\n", __func__, (unsigned long long)value);

  strela_semaphore_t* semaphore = (strela_semaphore_t*)base_semaphore;

  // 1. Advance the simulated host-side tracking value.
  iree_atomic_store(&semaphore->payload_value, value, iree_memory_order_release);

  // 2. Notify the hardware (if you had a real device-side timeline semaphore).
  // strela_hw_signal_semaphore(..., value);

  iree_async_semaphore_advance_timeline(base_semaphore, value, frontier);

  // 3. Wake up the async proactor.
  // Note: Depending on your exact IREE revision, the base `iree_async_semaphore_t`
  // might automatically resolve waiting nodes when this vtable hook returns `OK`,
  // or you might need to explicitly call a proactor wake-up function like:
  // iree_async_semaphore_advance(base_semaphore, value, frontier);

  return iree_ok_status();
}

static void
strela_semaphore_on_fail(iree_async_semaphore_t* semaphore, iree_status_code_t status_code) {
  printf("%s\n", __func__);
}

static const iree_hal_semaphore_vtable_t
strela_semaphore_vtable = {
  .async = {
    .destroy = strela_semaphore_destroy,
    .query = strela_semaphore_query,
    .signal = strela_semaphore_signal,
    .on_fail = strela_semaphore_on_fail,
  },
  .wait             = strela_semaphore_wait,
  .import_timepoint = strela_semaphore_import_timepoint,
  .export_timepoint = strela_semaphore_export_timepoint,
};
