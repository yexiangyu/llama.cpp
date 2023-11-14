#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define GGML_ACL_MAX_DEVICES 16

	GGML_API void ggml_acl_init(void);
	GGML_API void *ggml_acl_host_malloc(size_t size);
	GGML_API void ggml_acl_host_free(void *ptr);

	GGML_API bool ggml_acl_can_mul_mat(const struct ggml_tensor *src0, const struct ggml_tensor *src1, struct ggml_tensor *dst);
	GGML_API void ggml_acl_set_tensor_split(const float *tensor_split);
	GGML_API void ggml_acl_transform_tensor(void *data, struct ggml_tensor *tensor);
	GGML_API void ggml_acl_free_data(struct ggml_tensor *tensor);

	GGML_API void ggml_acl_assign_buffers(struct ggml_tensor *tensor);
	GGML_API void ggml_acl_assign_buffers_no_scratch(struct ggml_tensor *tensor);
	GGML_API void ggml_acl_assign_buffers_force_inplace(struct ggml_tensor *tensor);

	GGML_API void ggml_acl_assign_buffers_no_alloc(struct ggml_tensor *tensor);
	GGML_API void ggml_acl_assign_scratch_offset(struct ggml_tensor *tensor, size_t offset);
	GGML_API void ggml_acl_copy_to_device(struct ggml_tensor *tensor);

	GGML_API void ggml_acl_set_main_device(int main_device);
	GGML_API void ggml_acl_set_mul_mat_q(bool mul_mat_q);
	GGML_API void ggml_acl_set_scratch_size(size_t scratch_size);
	GGML_API void ggml_acl_free_scratch(void);
	GGML_API bool ggml_acl_compute_forward(struct ggml_compute_params *params, struct ggml_tensor *tensor);

	GGML_API int ggml_acl_get_device_count(void);
	GGML_API void ggml_acl_get_device_description(int device, char *description, size_t description_size);

	// backend API
	GGML_API ggml_backend_t ggml_backend_acl_init(void); // TODO: take a list of devices to use

#ifdef __cplusplus
}
#endif
