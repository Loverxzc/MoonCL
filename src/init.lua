local ffi = require("ffi")

ffi.cdef[[
    typedef void* cl_mem;
    typedef void* cl_event;
    typedef struct CustomKernelHandle CustomKernelHandle;
    typedef struct { float x, y, z, w; } mooncl_float4;

    int  mooncl_init(char* out_info, int max_len);
    void mooncl_cleanup();
    int  mooncl_is_ready();

    CustomKernelHandle* mooncl_compile_kernel(const char* source, const char* kernel_name, char* out_log, int max_log);
    CustomKernelHandle* mooncl_get_builtin_kernel(const char* kernel_name, char* out_log, int max_log);
    void                mooncl_free_kernel(CustomKernelHandle* handle);
    
    cl_mem   mooncl_create_buffer(size_t size);
    void     mooncl_free_buffer(cl_mem buf);
    int      mooncl_write_buffer(cl_mem buf, const void* data, size_t size);
    int      mooncl_read_buffer(cl_mem buf, void* data, size_t size);
    cl_event mooncl_read_buffer_async(cl_mem buf, void* data, size_t size);
    
    int      mooncl_set_arg_buffer(CustomKernelHandle* handle, int index, cl_mem buf);
    int      mooncl_set_arg_val(CustomKernelHandle* handle, int index, const void* val_ptr, size_t size);
    int      mooncl_run_kernel(CustomKernelHandle* handle, size_t global_size);
    cl_event mooncl_run_kernel_async(CustomKernelHandle* handle, size_t global_size);
    int      mooncl_run_kernel_nd(CustomKernelHandle* handle, int work_dim, const size_t* global_work_size);
    cl_event mooncl_run_kernel_nd_async(CustomKernelHandle* handle, int work_dim, const size_t* global_work_size);

    int      mooncl_event_is_done(cl_event evt);
    int      mooncl_event_wait(cl_event evt);
    void     mooncl_event_free(cl_event evt);
]]

---@class MoonCLNative
---@field mooncl_init fun(out_info: ffi.cdata*|string, max_len: integer): integer
---@field mooncl_cleanup fun()
---@field mooncl_is_ready fun(): integer
---@field mooncl_create_buffer fun(size: integer): ffi.cdata*
---@field mooncl_free_buffer fun(buf: ffi.cdata*)
---@field mooncl_write_buffer fun(buf: ffi.cdata*, data: ffi.cdata*|string, size: integer): integer
---@field mooncl_read_buffer fun(buf: ffi.cdata*, data: ffi.cdata*, size: integer): integer
---@field mooncl_read_buffer_async fun(buf: ffi.cdata*, data: ffi.cdata*, size: integer): ffi.cdata*
---@field mooncl_compile_kernel fun(source: string, kernel_name: string, out_log: ffi.cdata*, max_log: integer): ffi.cdata*
---@field mooncl_get_builtin_kernel fun(kernel_name: string, out_log: ffi.cdata*, max_log: integer): ffi.cdata*
---@field mooncl_free_kernel fun(handle: ffi.cdata*)
---@field mooncl_set_arg_buffer fun(handle: ffi.cdata*, index: integer, buf: ffi.cdata*): integer
---@field mooncl_set_arg_val fun(handle: ffi.cdata*, index: integer, val_ptr: ffi.cdata*, size: integer): integer
---@field mooncl_run_kernel fun(handle: ffi.cdata*, global_size: integer): integer
---@field mooncl_run_kernel_async fun(handle: ffi.cdata*, global_size: integer): ffi.cdata*
---@field mooncl_run_kernel_nd fun(handle: ffi.cdata*, work_dim: integer, global_work_size: ffi.cdata*): integer
---@field mooncl_run_kernel_nd_async fun(handle: ffi.cdata*, work_dim: integer, global_work_size: ffi.cdata*): ffi.cdata*
---@field mooncl_event_is_done fun(evt: ffi.cdata*): integer
---@field mooncl_event_wait fun(evt: ffi.cdata*): integer
---@field mooncl_event_free fun(evt: ffi.cdata*)

---@class MoonCLEvent
---@field handle ffi.cdata*|nil
local Event = {}
Event.__index = Event

---@class MoonCLBuffer
---@field handle ffi.cdata*|nil
---@field size integer
local Buffer = {}
Buffer.__index = Buffer

---@class MoonCLKernel
---@field handle ffi.cdata*|nil
---@field name string
local Kernel = {}
Kernel.__index = Kernel

local M = {}

---@type MoonCLNative|nil
local cl = nil

---@return MoonCLNative
local function get_dll()
    if not cl then
        cl = ffi.load(getWorkingDirectory() .. "\\lib\\mooncl\\mooncl.dll")
    end
    return cl
end

---@param handle ffi.cdata*
---@return MoonCLEvent
local function wrap_event(handle)
    local dll = get_dll()
    ffi.gc(handle, dll.mooncl_event_free)
    return setmetatable({ handle = handle }, Event)
end

---@return boolean
function Event:is_done()
    if not self.handle then return true end
    local dll = get_dll()
    return dll.mooncl_event_is_done(self.handle) == 1
end

---@return boolean
function Event:is_ready()
    return self:is_done()
end

---@return boolean
function Event:wait()
    if not self.handle then return false end
    local dll = get_dll()
    return dll.mooncl_event_wait(self.handle) == 1
end

function Event:free()
    if self.handle then
        local dll = get_dll()
        ffi.gc(self.handle, nil)
        dll.mooncl_event_free(self.handle)
        self.handle = nil
    end
end

---@return boolean success
---@return string info_or_error
function M.init()
    local dll = get_dll()
    local info_buf = ffi.new("char[2048]")
    local res = dll.mooncl_init(info_buf, 2048)

    local msg = ffi.string(info_buf)
    if res == 1 then
        return true, msg
    else
        return false, string.format("Код %d: %s", res, msg ~= "" and msg or "Неизвестная ошибка инициализации")
    end
end

---@return boolean
function M.is_ready()
    if not cl then return false end
    return cl.mooncl_is_ready() == 1
end

function M.cleanup()
    if cl then
        cl.mooncl_cleanup()
    end
end

---@param arg_idx integer
---@param x number|table|ffi.cdata*
---@param y? number
---@param z? number
---@param w? number
---@return boolean
function Kernel:set_float4(arg_idx, x, y, z, w)
    assert(self.handle ~= nil, "[MoonCL] Ядро уже освобождено!")
    
    local v
    if type(x) == "cdata" then
        return self:set_raw(arg_idx, x, 16)
    elseif type(x) == "table" then
        v = ffi.new("mooncl_float4", x[1] or x.x or 0, x[2] or x.y or 0, x[3] or x.z or 0, x[4] or x.w or 0)
    else
        v = ffi.new("mooncl_float4", x or 0, y or 0, z or 0, w or 0)
    end
    
    local dll = get_dll()
    return dll.mooncl_set_arg_val(self.handle, arg_idx, v, 16) == 1
end

---Создает массив float4 на стороне CPU для записи в буфер
---@param count integer
---@return ffi.cdata*
function M.new_float4_array(count)
    return ffi.new("mooncl_float4[?]", count)
end

---@param size_bytes integer
---@return MoonCLBuffer
function M.create_buffer(size_bytes)
    assert(M.is_ready(), "[MoonCL] GPU не инициализирован!")
    local dll = get_dll()
    local mem = dll.mooncl_create_buffer(size_bytes)
    assert(mem ~= nil, "[MoonCL] Не удалось выделить буфер в VRAM!")
    
    ffi.gc(mem, dll.mooncl_free_buffer)

    return setmetatable({ handle = mem, size = size_bytes }, Buffer)
end

---@param cdata ffi.cdata*|string
---@param size? integer
---@return boolean
function Buffer:write(cdata, size)
    assert(self.handle ~= nil, "[MoonCL] Попытка записи в освобожденный буфер!")
    local dll = get_dll()
    return dll.mooncl_write_buffer(self.handle, cdata, size or self.size) == 1
end

---@param cdata ffi.cdata*
---@param size? integer
---@return boolean
function Buffer:read(cdata, size)
    assert(self.handle ~= nil, "[MoonCL] Попытка чтения из освобожденного буфера!")
    local dll = get_dll()
    return dll.mooncl_read_buffer(self.handle, cdata, size or self.size) == 1
end

---@param cdata ffi.cdata*
---@param size? integer
---@return MoonCLEvent|nil
function Buffer:read_async(cdata, size)
    assert(self.handle ~= nil, "[MoonCL] Попытка чтения из освобожденного буфера!")
    local dll = get_dll()
    local evt = dll.mooncl_read_buffer_async(self.handle, cdata, size or self.size)
    if evt == nil then return nil end
    return wrap_event(evt)
end

function Buffer:free()
    if self.handle then
        local dll = get_dll()
        ffi.gc(self.handle, nil)
        dll.mooncl_free_buffer(self.handle)
        self.handle = nil
    end
end

---@param source string
---@param kernel_name string
---@return MoonCLKernel|nil kernel
---@return string|nil error_log
function M.compile(source, kernel_name)
    assert(M.is_ready(), "[MoonCL] GPU не инициализирован!")
    local dll = get_dll()
    local log_buf = ffi.new("char[4096]")
    local handle = dll.mooncl_compile_kernel(source, kernel_name, log_buf, 4096)

    if handle == nil then
        return nil, ffi.string(log_buf)
    end

    ffi.gc(handle, dll.mooncl_free_kernel)

    return setmetatable({ handle = handle, name = kernel_name }, Kernel), nil
end

---@param kernel_name string
---@return MoonCLKernel|nil kernel
---@return string|nil error_log
function M.get_kernel(kernel_name)
    assert(M.is_ready(), "[MoonCL] GPU не инициализирован!")
    local dll = get_dll()
    local log_buf = ffi.new("char[4096]")
    local handle = dll.mooncl_get_builtin_kernel(kernel_name, log_buf, 4096)

    if handle == nil then
        return nil, ffi.string(log_buf)
    end

    ffi.gc(handle, dll.mooncl_free_kernel)

    return setmetatable({ handle = handle, name = kernel_name }, Kernel), nil
end

---@param arg_idx integer
---@param buffer_obj MoonCLBuffer|ffi.cdata*
---@return boolean
function Kernel:set_buffer(arg_idx, buffer_obj)
    assert(self.handle ~= nil, "[MoonCL] Ядро уже освобождено!")
    local dll = get_dll()

    local h = buffer_obj
    if type(buffer_obj) == "table" then
        h = buffer_obj.handle
    end

    ---@cast h ffi.cdata*
    assert(h ~= nil, "[MoonCL] Передан пустой или освобожденный буфер!")

    return dll.mooncl_set_arg_buffer(self.handle, arg_idx, h) == 1
end

---@param arg_idx integer
---@param val number
---@return boolean
function Kernel:set_float(arg_idx, val)
    assert(self.handle ~= nil, "[MoonCL] Ядро уже освобождено!")
    local dll = get_dll()
    local v = ffi.new("float[1]", val)
    return dll.mooncl_set_arg_val(self.handle, arg_idx, v, 4) == 1
end

---@param arg_idx integer
---@param val integer
---@return boolean
function Kernel:set_int(arg_idx, val)
    assert(self.handle ~= nil, "[MoonCL] Ядро уже освобождено!")
    local dll = get_dll()
    local v = ffi.new("int[1]", val)
    return dll.mooncl_set_arg_val(self.handle, arg_idx, v, 4) == 1
end

---@param arg_idx integer
---@param cdata_ptr ffi.cdata*
---@param type_size integer
---@return boolean
function Kernel:set_raw(arg_idx, cdata_ptr, type_size)
    assert(self.handle ~= nil, "[MoonCL] Ядро уже освобождено!")
    local dll = get_dll()
    return dll.mooncl_set_arg_val(self.handle, arg_idx, cdata_ptr, type_size) == 1
end

---@param global_size integer
---@return boolean
function Kernel:run(global_size)
    assert(self.handle ~= nil, "[MoonCL] Ядро уже освобождено!")
    local dll = get_dll()
    return dll.mooncl_run_kernel(self.handle, global_size) == 1
end

---@param global_size integer
---@return MoonCLEvent|nil
function Kernel:run_async(global_size)
    assert(self.handle ~= nil, "[MoonCL] Ядро уже освобождено!")
    local dll = get_dll()
    local evt = dll.mooncl_run_kernel_async(self.handle, global_size)
    if evt == nil then return nil end
    return wrap_event(evt)
end

---@param work_dim integer
---@param sizes_table integer[]
---@return boolean
function Kernel:run_nd(work_dim, sizes_table)
    assert(self.handle ~= nil, "[MoonCL] Ядро уже освобождено!")
    local dll = get_dll()
    local c_sizes = ffi.new("size_t[?]", work_dim, sizes_table)
    return dll.mooncl_run_kernel_nd(self.handle, work_dim, c_sizes) == 1
end

---@param work_dim integer
---@param sizes_table integer[]
---@return MoonCLEvent|nil
function Kernel:run_nd_async(work_dim, sizes_table)
    assert(self.handle ~= nil, "[MoonCL] Ядро уже освобождено!")
    local dll = get_dll()
    local c_sizes = ffi.new("size_t[?]", work_dim, sizes_table)
    local evt = dll.mooncl_run_kernel_nd_async(self.handle, work_dim, c_sizes)
    if evt == nil then return nil end
    return wrap_event(evt)
end

function Kernel:free()
    if self.handle then
        local dll = get_dll()
        ffi.gc(self.handle, nil)
        dll.mooncl_free_kernel(self.handle)
        self.handle = nil
    end
end

return M