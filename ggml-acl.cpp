#include "ggml-acl.h"
#include <iostream>
#include <cstdlib>
#include "acl/acl.h"

#define acl_rc(err)                                                                                                                        \
	do                                                                                                                                     \
	{                                                                                                                                      \
		if (err != ACL_SUCCESS)                                                                                                            \
		{                                                                                                                                  \
			std::cerr << "acl error, function=" << __func__ << ",line=" << __LINE__ << ",file=" << __FILE__ << ",rc=" << err << std::endl; \
			exit(err);                                                                                                                     \
		}                                                                                                                                  \
	} while (false)

static uint32_t g_device_count = 0;
static float g_tensor_split[GGML_ACL_MAX_DEVICES] = {0.0};

static int64_t get_row_rounding(ggml_type type)
{
	switch (type)
	{
		// how about f32?
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q8_0:
		return 64;
	case GGML_TYPE_F16:
		return 1;
	case GGML_TYPE_Q2_K:
	case GGML_TYPE_Q3_K:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_Q6_K:
		return 64;
	default:
		GGML_ASSERT(false);
	}
}

void set_current_device(uint32_t id)
{
	acl_rc(aclrtSetDevice(id));
}

void ggml_acl_init(void)
{
	static bool initialized = false;
	if (initialized)
	{
		return;
	}
	acl_rc(aclInit(nullptr));
	acl_rc(aclrtGetDeviceCount(&g_device_count));
	size_t total_device_mem = 0;
	for (auto i = 0; i < g_device_count; i++)
	{
		size_t free = 0;
		size_t total = 0;
		acl_rc(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
		total_device_mem += total;
		g_tensor_split[i] = total_device_mem;
	}
	for (auto i = 0; i < g_device_count; i++)
	{
		g_tensor_split[i] /= total_device_mem;
	}
	initialized = true;
}

void *ggml_acl_host_malloc(size_t size)
{
	void *ptr = nullptr;
	acl_rc(aclrtMallocHost(&ptr, size));
	return ptr;
}

void ggml_acl_host_free(void *ptr)
{
	acl_rc(aclrtFreeHost(ptr));
}

bool ggml_acl_can_mul_mat(const struct ggml_tensor *src0, const struct ggml_tensor *src1, struct ggml_tensor *dst)
{
	return true;
}

void ggml_acl_set_tensor_split(const float *tensor_split)
{
	return;
}

struct tensor_in_acl
{
};

void ggml_acl_transform_tensor(void *data, struct ggml_tensor *tensor)
{
	const int64_t nrows = ggml_nrows(tensor);

	auto extra = new tensor_in_acl();

	for (auto id = 0; id < g_device_count; id++)
	{
		set_current_device(id);
		int64_t row_start, row_end;
		auto rounding = get_row_rounding(tensor->type);
		if (g_device_count == 1)
		{
			row_start = 0;
			row_end = nrows;
		}
		else
		{
			row_start = id == 0 ? 0 : nrows * g_tensor_split[id];
			row_start -= row_start % rounding;
		}

		if (id == g_device_count - 1)
		{
			row_end = nrows;
		}
		else
		{
			row_end = nrows * g_tensor_split[id + 1];
			row_end -= row_end % rounding;
		}

		auto nrows_split = row_end - row_start;

		if (nrows_split == 0)
		{
			continue;
		}
	}

	return;
}

void ggml_acl_free_data(struct ggml_tensor *tensor)
{
	return;
}

void ggml_acl_assign_buffers(struct ggml_tensor *tensor)
{
	return;
}

void ggml_acl_assign_buffers_no_scratch(struct ggml_tensor *tensor)
{
	return;
}

void ggml_acl_assign_buffers_force_inplace(struct ggml_tensor *tensor)
{
	return;
}

void ggml_acl_assign_buffers_no_alloc(struct ggml_tensor *tensor)
{
	return;
}
void ggml_acl_assign_scratch_offset(struct ggml_tensor *tensor, size_t offset)
{
	return;
}
void ggml_acl_copy_to_device(struct ggml_tensor *tensor)
{
	return;
}

void ggml_acl_set_main_device(int main_device)
{
	return;
}
void ggml_acl_set_mul_mat_q(bool mul_mat_q)
{
	return;
}
void ggml_acl_set_scratch_size(size_t scratch_size)
{
	return;
}
void ggml_acl_free_scratch(void)
{
	return;
}

bool ggml_acl_compute_forward(struct ggml_compute_params *params, struct ggml_tensor *tensor)
{
	return false;
}

int ggml_acl_get_device_count(void);
void ggml_acl_get_device_description(int device, char *description, size_t description_size);

ggml_backend_t ggml_backend_acl_init(void); // TODO: take a list of devices to use