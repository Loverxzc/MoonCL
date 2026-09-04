# MoonCL - параллельные вычисления на видеоядре прямо в Lua скрипте!

![Version](https://img.shields.io/badge/version-0.1.0--pre--release-orange.svg)
![Platform](https://img.shields.io/badge/platform-MoonLoader-blue.svg)
![OpenCL](https://img.shields.io/badge/OpenCL-1.2-green.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey.svg)

[![Скачать MoonCL](https://img.shields.io/badge/Скачать_MoonCL-v0.1.0-2ea44f?style=for-the-badge&logo=github)](https://github.com/Loverxzc/MoonCL/releases)

**MoonCL** - это легковесный OpenCL 1.2 мост для **MoonLoader**, позволяющий перенести массивные параллельные геометрические, матричные и пространственные расчеты прямо на GPU.

Библиотека создана для решения специфических вычислительных задач: кастомные рейкасты, проверка сотен хитбоксов, пространственная кластеризация объектов, обработка геометрии и симуляции без блокировки игрового цикла.
Эта библиотека не для каждодневного скриптинга, большинство обычных задач лучше делать через обычные циклы или через FFI.

---

## Оглавление
- [Зачем это нужно?](#зачем-это-нужно)
- [Особенности архитектуры](#особенности-архитектуры)
- [Установка](#установка)
- [Встроенные ядра (Built-in Kernels)](#встроенные-ядра-built-in-kernels)
- [Быстрый старт (Quick Start)](#быстрый-старт-quick-start)
- [Справочник API (Lua)](#справочник-api-lua)
  - [Модуль mooncl](#модуль-mooncl)
  - [Объект Buffer](#объект-buffer)
  - [Объект Kernel](#объект-kernel)
  - [Объект Event (Асинхронность)](#объект-event-асинхронность)
- [Кастомные ядра (JIT на лету)](#кастомные-ядра-jit-на-лету)
- [Нюансы и ограничения пред-релизов](#нюансы-и-ограничения-пред-релизов)
- [Лицензия](#лицензия)

---

## Зачем это нужно?

MoonLoader работают в одном главном потоке. При 60 FPS на весь кадр выделяется всего **~16.6 миллисекунд**. 

Если ваш Lua-скрипт выполняет тяжелую математику (проверка луча с сотнями bounding box'ов, пересчет матриц для сотен точек, расчет дистанций до тысяч маркеров/игроков), LuaJIT неизбежно превышает бюджет кадра, вызывая микрофризы.

**MoonCL решает эту проблему двумя путями:**
1. **Массивный параллелизм:** GPU обрабатывает десятки тысяч точек за микросекунды благодаря сотням вычислительных ядер.
2. **Асинхронные очереди (Non-blocking):** Вы отправляете задачу на GPU, игра продолжает рендериться со стабильным FPS, а скрипт проверяет готовность данных через `Event:is_done()` в стандартном игровом цикле.

---

## Особенности архитектуры

* **Встроенные ядра без внешних файлов:** Стандартные OpenCL-ядра (`kernel.cl`) вшиты прямо в бинарник `.dll` на этапе компиляции через GNU ассемблер (`.incbin`).
* **Автоматический сборщик мусора:** Все OpenCL-дескрипторы (буферы, события, скомпилированные ядра) привязаны к `ffi.gc`. Память в VRAM освобождается сборщиком мусора LuaJIT, даже если скрипт был экстренно перезагружен по `Ctrl+R`.
* **Безопасная повторная инициализация:** Встроенный счетчик ссылок (`g_ref_count`) позволяет нескольким скриптам в одной игре безопасно делить один OpenCL-контекст.

---

## Установка

1. Скопируйте файлы в папку вашей игры:
   ```text
   moonloader/
   └── lib/
       └── mooncl/
           ├── init.lua     <-- Lua обертка
           └── mooncl.dll   <-- бинарник
    ```
2. Подключите в начале вашего скрипта:
   local mooncl = require("mooncl")
---

## Встроенные ядра (Built-in Kernels)

В библиотеку уже вшиты базовые ядра, скомпилированные с флагом -cl-fast-relaxed-math:

| Имя ядра | Аргументы | Описание |
| :--- | :--- | :--- |
| mat4_transform_points | in_points, mat, out_points, count | Умножение массива точек float4 на матрицу 4x4. Подходит для массовой трансформации вершин и костей. |
| ray_aabb_intersect | ray_origin, ray_dir_inv, aabb_mins, aabb_maxs, out_t, count | Проверка пересечения луча с массивом AABB (bounding box). Возвращает дистанцию t или -1.0 при промахе. |
| batch_distance | in_points, target, out_distances, count | Пакетный расчет евклидова расстояния от массива точек до целевой точки target.xyz. |
| batch_spatial_filter | in_points, target, radius_sq, out_mask, count | Проверка радиуса (dist^2 <= radius^2). Записывает маску 1 (внутри) или 0 (снаружи). |
| batch_clamp | in_vals, out_vals, min_val, max_val, count | Ограничение значений массива в заданном диапазоне [min_val, max_val]. |
| batch_lerp | a, b, t, out_vals, count | Линейная интерполяция двух массивов чисел (mix(a, b, t)). |

---

## Быстрый старт (Quick Start)

Пример асинхронного расчета дистанций от 10 000 точек до игрока без блокировки игрового кадра:

```lua
local mooncl = require("mooncl")
local ffi = require("ffi")

function main()
    while not isSampAvailable() do wait(100) end

    -- Инициализация OpenCL
    local ok, dev_name = mooncl.init()
    if not ok then
        print("[MoonCL] Ошибка: " .. dev_name)
        return
    end
    print("[MoonCL] Устройство: " .. dev_name)

    -- Подготовка точек на стороне CPU
    local COUNT = 10000
    local points = mooncl.new_float4_array(COUNT)
    for i = 0, COUNT - 1 do
        points[i].x = i * 1.5
        points[i].y = i * 0.5
        points[i].z = 0.0
        points[i].w = 1.0
    end

    -- Создание буферов в видеопамяти
    local in_buf = mooncl.create_buffer(COUNT * 16)      -- sizeof(float4) = 16
    local out_buf = mooncl.create_buffer(COUNT * 4)      -- sizeof(float) = 4

    -- Запись данных в GPU
    in_buf:write(points)

    -- Получение ядра и бинд аргументов
    local kernel, err = mooncl.get_kernel("batch_distance")
    assert(kernel, err)

    kernel:set_buffer(0, in_buf)
    kernel:set_float4(1, 0.0, 0.0, 0.0, 0.0) -- целевая точка
    kernel:set_buffer(2, out_buf)
    kernel:set_int(3, COUNT)

    -- Асинхронный запуск расчета
    local run_event = kernel:run_async(COUNT)

    -- Ждем завершения расчета
    while not run_event:is_done() do
        wait(0)
    end

    -- Асинхронное чтение результатов
    local results = ffi.new("float[?]", COUNT)
    local read_event = out_buf:read_async(results)

    while not read_event:is_done() do
        wait(0)
    end

    print(string.format("[MoonCL] Дистанция до точки 500: %.2f", results[500]))

    -- Освобождение ресурсов (память также подчищается сборщиком через ffi.gc)
    in_buf:free()
    out_buf:free()
    kernel:free()
end
```

---

## Справочник API (Lua)

### Модуль `mooncl`

* `mooncl.init()` $\to$ `boolean success, string info_or_error`  
  Ищет доступный GPU (при отсутствии падает на CPU-устройство), создает контекст, очередь команд и компилирует базовые ядра. Возвращает статус и название найденного устройства.
* `mooncl.is_ready()` $\to$ `boolean`  
  Возвращает `true`, если OpenCL инициализирован и готов к приему команд.
* `mooncl.cleanup()`  
  Уменьшает счетчик ссылок и освобождает ресурсы OpenCL (программу, очередь, контекст), когда счетчик достигает `0`.
* `mooncl.create_buffer(size_bytes: integer)` $\to$ `MoonCLBuffer`  
  Выделяет буфер в видеопамяти (`CL_MEM_READ_WRITE`).
* `mooncl.new_float4_array(count: integer)` $\to$ `ffi.cdata*`  
  Хелпер для создания массива структур `mooncl_float4[count]` в системной памяти (RAM).
* `mooncl.get_kernel(kernel_name: string)` $\to$ `MoonCLKernel|nil, string|nil`  
  Возвращает хэндл одного из встроенных ядер.
* `mooncl.compile(source_code: string, kernel_name: string)` $\to$ `MoonCLKernel|nil, string|nil`  
  Компилирует произвольный код ядра OpenCL на лету. При ошибке возвращает лог сборщика.

---

### Объект `Buffer`

* `buffer:write(cdata: ffi.cdata*|string, size?: integer)` $\to$ `boolean`  
  Блокирующая синхронная запись данных из RAM в видеопамять VRAM.
* `buffer:read(cdata: ffi.cdata*, size?: integer)` $\to$ `boolean`  
  Блокирующее чтение данных из VRAM в RAM (останавливает поток игры до окончания передачи).
* `buffer:read_async(cdata: ffi.cdata*, size?: integer)` $\to$ `MoonCLEvent|nil`  
  **Рекомендуется.** Неблокирующее чтение данных. Запускает передачу и возвращает объект события.
* `buffer:free()`  
  Освобождает буфер памяти на GPU вручную.

---

### Объект `Kernel`

* `kernel:set_buffer(index: integer, buffer: MoonCLBuffer|ffi.cdata*)` $\to$ `boolean`  
  Привязывает буфер памяти к аргументу с индексом `index` (нумерация с `0`).
* `kernel:set_float(index: integer, number: number)` $\to$ `boolean`  
  Передает скалярный аргумент типа `float`.
* `kernel:set_int(index: integer, integer: integer)` $\to$ `boolean`  
  Передает аргумент типа `int`.
* `kernel:set_float4(index: integer, x: number|table|ffi.cdata*, y?: number, z?: number, w?: number)` $\to$ `boolean`  
  Передает векторный аргумент типа `float4`. Принимает 4 числа, массив `{x, y, z, w}` или готовый FFI `cdata`-объект.
* `kernel:set_raw(index: integer, cdata_ptr: ffi.cdata*, size_bytes: integer)` $\to$ `boolean`  
  Передает произвольную C-структуру по значению.
* `kernel:run(global_size: integer)` $\to$ `boolean`  
  Блокирующий 1D запуск ядра.
* `kernel:run_async(global_size: integer)` $\to$ `MoonCLEvent|nil`  
  **Рекомендуется.** Неблокирующий 1D запуск ядра. Возвращает событие выполнения.
* `kernel:run_nd(dimensions: integer, sizes_table: integer[])` $\to$ `boolean`  
  Блокирующий запуск с кастомной размерностью сетки (1D, 2D или 3D). Таблица размеров передается массивом (например, `{1024, 1024}`).
* `kernel:run_nd_async(dimensions: integer, sizes_table: integer[])` $\to$ `MoonCLEvent|nil`  
  Неблокирующий многомерный запуск.
* `kernel:free()`  
  Освобождает хэндл ядра вручную.

---

### Объект `Event` (Асинхронность)

* `event:is_done()` $\to$ `boolean`  
  Проверяет, завершилась ли операция на видеокарте. Не блокирует игровой кадр.
* `event:wait()` $\to$ `boolean`  
  Принудительно останавливает текущий поток до завершения события.
* `event:free()`  
  Освобождает дескриптор события вручную.

---

## Кастомные ядра (JIT на лету)

Если возможностей встроенных ядер недостаточно, можно компилировать свой код прямо во время работы скрипта:

```lua
local source = [[
__kernel void vector_add(__global const float* a, __global const float* b, __global float* c, int count) {
    int id = get_global_id(0);
    if (id >= count) return;
    c[id] = a[id] + b[id];
}
]]

local kernel, log = mooncl.compile(source, "vector_add")
if not kernel then
    print("[MoonCL] Ошибка компиляции ядра:\n" .. log)
    return
end

-- Использование идентично встроенным ядрам
```

---

## Нюансы и ограничения пред-релизов.

1. **Выравнивание памяти `float4`:**  
   В OpenCL структуры `float4` имеют строгое выравнивание по **16 байт**. При формировании буферов на стороне Lua необходимо учитывать эти отступы (используйте `mooncl.new_float4_array`).
2. **Локальный размер группы (`local_work_size`):**  
   В пред-релизных версиях локальный размер рабочей группы передается как `NULL`. Драйвер видеокарты выбирает его самостоятельно. Использование локальной памяти `__local` внутри ядер пока не поддерживается оптимальным образом.
3. **Накладные расходы шины PCIe:**  
   Пересылка данных между RAM и VRAM требует времени. Отправлять на GPU массивы размером меньше 500–1000 элементов не имеет практического смысла — затраты на передачу перекроют выигрыш от распараллеливания.
4. **Асинхронность:**  
   Синхронные методы (`:run()`, `:read()`) при больших объемах данных могут остановить поток на несколько миллисекунд. Для сохранения FPS рекомендуется использовать `:run_async()` и `:read_async()` с проверкой статуса `:is_done()` в игровом цикле.

---

## Лицензия

[Apache 2.0](LICENSE)