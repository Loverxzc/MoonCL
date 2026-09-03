#include <cstdio>
#include <cstring>
#include <cstdlib>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#if defined(_WIN32)
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

__asm__(
    ".section .rodata\n"
    ".global kernel_source_code\n"
    ".global _kernel_source_code\n"
    "kernel_source_code:\n"
    "_kernel_source_code:\n"
    ".incbin \"src/kernel.cl\"\n"
    ".byte 0\n"
    ".section .text\n"
);

extern "C" const char kernel_source_code[];

static cl_context       g_context   = nullptr;
static cl_command_queue g_queue     = nullptr;
static cl_program       g_program   = nullptr;
static cl_device_id     g_device    = nullptr;
static int              g_ref_count = 0;

struct CustomKernelHandle {
    cl_program program;
    cl_kernel  kernel;
};

extern "C" {

EXPORT int mooncl_init(char* out_info, int max_len) {
    if (out_info && max_len > 0) out_info[0] = '\0';

    if (g_context) {
        g_ref_count++;
        if (out_info && max_len > 0) {
            clGetDeviceInfo(g_device, CL_DEVICE_NAME, max_len, out_info, NULL);
        }
        return 1;
    }

    cl_uint num_platforms = 0;
    cl_int res = clGetPlatformIDs(0, NULL, &num_platforms);
    if (res != CL_SUCCESS || num_platforms == 0) {
        if (out_info && max_len > 0) snprintf(out_info, max_len, "OpenCL platforms not found");
        return -1;
    }

    cl_platform_id platforms[16];
    if (num_platforms > 16) num_platforms = 16;
    clGetPlatformIDs(num_platforms, platforms, NULL);

    g_device = NULL;
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_dev = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &g_device, &num_dev) == CL_SUCCESS && num_dev > 0)
            break;
    }

    if (!g_device) {
        for (cl_uint i = 0; i < num_platforms; i++) {
            cl_uint num_dev = 0;
            if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1, &g_device, &num_dev) == CL_SUCCESS && num_dev > 0)
                break;
        }
    }

    if (!g_device) {
        if (out_info && max_len > 0) snprintf(out_info, max_len, "OpenCL device not found");
        return -2;
    }

    cl_int err;
    g_context = clCreateContext(NULL, 1, &g_device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !g_context) {
        if (out_info && max_len > 0) snprintf(out_info, max_len, "Failed to create context: %d", err);
        return -3;
    }

    g_queue = clCreateCommandQueue(g_context, g_device, 0, &err);
    if (err != CL_SUCCESS || !g_queue) {
        if (out_info && max_len > 0) snprintf(out_info, max_len, "Failed to create queue: %d", err);
        clReleaseContext(g_context);
        g_context = nullptr;
        return -4;
    }

    const char* src = kernel_source_code;
    g_program = clCreateProgramWithSource(g_context, 1, &src, NULL, &err);
    if (err == CL_SUCCESS && g_program) {
        clBuildProgram(g_program, 1, &g_device, "-cl-fast-relaxed-math", NULL, NULL);
    }

    if (out_info && max_len > 0) {
        clGetDeviceInfo(g_device, CL_DEVICE_NAME, max_len, out_info, NULL);
    }

    g_ref_count = 1;
    return 1;
}

EXPORT void mooncl_cleanup() {
    if (g_ref_count > 1) {
        g_ref_count--;
        return;
    }

    if (g_program) { clReleaseProgram(g_program);     g_program = nullptr; }
    if (g_queue)   { clReleaseCommandQueue(g_queue);   g_queue   = nullptr; }
    if (g_context) { clReleaseContext(g_context);     g_context = nullptr; }

    g_ref_count = 0;
}

EXPORT int mooncl_is_ready() {
    return (g_context != nullptr) ? 1 : 0;
}

EXPORT CustomKernelHandle* mooncl_compile_kernel(const char* source, const char* kernel_name, char* out_log, int max_log) {
    if (out_log && max_log > 0) out_log[0] = '\0';
    if (!g_context) return nullptr;

    cl_int err;
    cl_program prog = clCreateProgramWithSource(g_context, 1, &source, NULL, &err);
    if (err != CL_SUCCESS || !prog) {
        if (out_log && max_log > 0) snprintf(out_log, max_log, "Source Error: %d", err);
        return nullptr;
    }

    err = clBuildProgram(prog, 1, &g_device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        if (out_log && max_log > 0) clGetProgramBuildInfo(prog, g_device, CL_PROGRAM_BUILD_LOG, max_log, out_log, NULL);
        clReleaseProgram(prog);
        return nullptr;
    }

    cl_kernel krn = clCreateKernel(prog, kernel_name, &err);
    if (err != CL_SUCCESS || !krn) {
        if (out_log && max_log > 0) snprintf(out_log, max_log, "Kernel '%s' not found: %d", kernel_name, err);
        clReleaseProgram(prog);
        return nullptr;
    }

    CustomKernelHandle* handle = (CustomKernelHandle*)malloc(sizeof(CustomKernelHandle));
    if (!handle) {
        clReleaseKernel(krn);
        clReleaseProgram(prog);
        return nullptr;
    }

    handle->program = prog;
    handle->kernel = krn;
    return handle;
}

EXPORT CustomKernelHandle* mooncl_get_builtin_kernel(const char* kernel_name, char* out_log, int max_log) {
    if (out_log && max_log > 0) out_log[0] = '\0';
    if (!g_context || !g_program) return nullptr;

    cl_int err;
    cl_kernel krn = clCreateKernel(g_program, kernel_name, &err);
    if (err != CL_SUCCESS || !krn) {
        if (out_log && max_log > 0) snprintf(out_log, max_log, "Builtin kernel '%s' not found: %d", kernel_name, err);
        return nullptr;
    }

    CustomKernelHandle* handle = (CustomKernelHandle*)malloc(sizeof(CustomKernelHandle));
    if (!handle) {
        clReleaseKernel(krn);
        return nullptr;
    }

    handle->program = nullptr;
    handle->kernel = krn;
    return handle;
}

EXPORT void mooncl_free_kernel(CustomKernelHandle* handle) {
    if (!handle) return;
    if (handle->kernel)  clReleaseKernel(handle->kernel);
    if (handle->program) clReleaseProgram(handle->program);
    free(handle);
}

EXPORT cl_mem mooncl_create_buffer(size_t size) {
    if (!g_context || size == 0) return nullptr;
    return clCreateBuffer(g_context, CL_MEM_READ_WRITE, size, nullptr, nullptr);
}

EXPORT void mooncl_free_buffer(cl_mem buf) {
    if (buf) clReleaseMemObject(buf);
}

EXPORT int mooncl_write_buffer(cl_mem buf, const void* data, size_t size) {
    if (!g_queue || !buf || !data || size == 0) return 0;
    return clEnqueueWriteBuffer(g_queue, buf, CL_TRUE, 0, size, data, 0, nullptr, nullptr) == CL_SUCCESS;
}

EXPORT int mooncl_read_buffer(cl_mem buf, void* data, size_t size) {
    if (!g_queue || !buf || !data || size == 0) return 0;
    return clEnqueueReadBuffer(g_queue, buf, CL_TRUE, 0, size, data, 0, nullptr, nullptr) == CL_SUCCESS;
}

EXPORT cl_event mooncl_read_buffer_async(cl_mem buf, void* data, size_t size) {
    if (!g_queue || !buf || !data || size == 0) return nullptr;
    cl_event evt = nullptr;
    cl_int err = clEnqueueReadBuffer(g_queue, buf, CL_FALSE, 0, size, data, 0, nullptr, &evt);
    if (err != CL_SUCCESS) return nullptr;
    clFlush(g_queue);
    return evt;
}

EXPORT int mooncl_set_arg_buffer(CustomKernelHandle* handle, int index, cl_mem buf) {
    if (!handle || !handle->kernel) return 0;
    return clSetKernelArg(handle->kernel, index, sizeof(cl_mem), &buf) == CL_SUCCESS;
}

EXPORT int mooncl_set_arg_val(CustomKernelHandle* handle, int index, const void* val_ptr, size_t size) {
    if (!handle || !handle->kernel) return 0;
    return clSetKernelArg(handle->kernel, index, size, val_ptr) == CL_SUCCESS;
}

EXPORT int mooncl_run_kernel(CustomKernelHandle* handle, size_t global_size) {
    if (!g_queue || !handle || !handle->kernel || global_size == 0) return 0;
    return clEnqueueNDRangeKernel(g_queue, handle->kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr) == CL_SUCCESS;
}

EXPORT cl_event mooncl_run_kernel_async(CustomKernelHandle* handle, size_t global_size) {
    if (!g_queue || !handle || !handle->kernel || global_size == 0) return nullptr;
    cl_event evt = nullptr;
    cl_int err = clEnqueueNDRangeKernel(g_queue, handle->kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, &evt);
    if (err != CL_SUCCESS) return nullptr;
    clFlush(g_queue);
    return evt;
}

EXPORT int mooncl_run_kernel_nd(CustomKernelHandle* handle, int work_dim, const size_t* global_work_size) {
    if (!g_queue || !handle || !handle->kernel || work_dim < 1 || work_dim > 3 || !global_work_size) return 0;
    return clEnqueueNDRangeKernel(g_queue, handle->kernel, work_dim, nullptr, global_work_size, nullptr, 0, nullptr, nullptr) == CL_SUCCESS;
}

EXPORT cl_event mooncl_run_kernel_nd_async(CustomKernelHandle* handle, int work_dim, const size_t* global_work_size) {
    if (!g_queue || !handle || !handle->kernel || work_dim < 1 || work_dim > 3 || !global_work_size) return nullptr;
    cl_event evt = nullptr;
    cl_int err = clEnqueueNDRangeKernel(g_queue, handle->kernel, work_dim, nullptr, global_work_size, nullptr, 0, nullptr, &evt);
    if (err != CL_SUCCESS) return nullptr;
    clFlush(g_queue);
    return evt;
}

EXPORT int mooncl_event_is_done(cl_event evt) {
    if (!evt) return 0;
    cl_int status = 0;
    cl_int err = clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, nullptr);
    if (err != CL_SUCCESS) return -1;
    return (status == CL_COMPLETE) ? 1 : 0;
}

EXPORT int mooncl_event_wait(cl_event evt) {
    if (!evt) return 0;
    return clWaitForEvents(1, &evt) == CL_SUCCESS ? 1 : 0;
}

EXPORT void mooncl_event_free(cl_event evt) {
    if (evt) clReleaseEvent(evt);
}

}