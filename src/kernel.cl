__kernel void mat4_transform_points(
    __global const float4* in_points,
    __constant const float* mat,
    __global float4* out_points,
    int count)
{
    int id = get_global_id(0);
    if (id >= count) return;

    float4 p = in_points[id];
    float4 res;
    res.x = mat[0]  * p.x + mat[4]  * p.y + mat[8]  * p.z + mat[12] * p.w;
    res.y = mat[1]  * p.x + mat[5]  * p.y + mat[9]  * p.z + mat[13] * p.w;
    res.z = mat[2]  * p.x + mat[6]  * p.y + mat[10] * p.z + mat[14] * p.w;
    res.w = mat[3]  * p.x + mat[7]  * p.y + mat[11] * p.z + mat[15] * p.w;

    out_points[id] = res;
}

__kernel void ray_aabb_intersect(
    float4 ray_origin,
    float4 ray_dir_inv,
    __global const float4* aabb_mins,
    __global const float4* aabb_maxs,
    __global float* out_t,
    int count)
{
    int id = get_global_id(0);
    if (id >= count) return;

    float4 bmin = aabb_mins[id];
    float4 bmax = aabb_maxs[id];

    float4 t1 = (bmin - ray_origin) * ray_dir_inv;
    float4 t2 = (bmax - ray_origin) * ray_dir_inv;

    float4 tmin4 = fmin(t1, t2);
    float4 tmax4 = fmax(t1, t2);

    float tmin = fmax(fmax(tmin4.x, tmin4.y), tmin4.z);
    float tmax = fmin(fmin(tmax4.x, tmax4.y), tmax4.z);

    if (tmax >= fmax(tmin, 0.0f)) {
        out_t[id] = tmin > 0.0f ? tmin : 0.0f;
    } else {
        out_t[id] = -1.0f;
    }
}

__kernel void batch_distance(
    __global const float4* in_points,
    float4 target,
    __global float* out_distances,
    int count)
{
    int id = get_global_id(0);
    if (id >= count) return;

    out_distances[id] = distance(in_points[id].xyz, target.xyz);
}

__kernel void batch_spatial_filter(
    __global const float4* in_points,
    float4 target,
    float radius_sq,
    __global int* out_mask,
    int count)
{
    int id = get_global_id(0);
    if (id >= count) return;

    float4 diff = in_points[id] - target;
    float d2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    out_mask[id] = (d2 <= radius_sq) ? 1 : 0;
}

__kernel void batch_clamp(
    __global const float* in_vals,
    __global float* out_vals,
    float min_val,
    float max_val,
    int count)
{
    int id = get_global_id(0);
    if (id >= count) return;

    out_vals[id] = clamp(in_vals[id], min_val, max_val);
}

__kernel void batch_lerp(
    __global const float* a,
    __global const float* b,
    float t,
    __global float* out_vals,
    int count)
{
    int id = get_global_id(0);
    if (id >= count) return;

    out_vals[id] = mix(a[id], b[id], t);
}