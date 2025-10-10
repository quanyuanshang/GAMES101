// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(int x, int y, const Vector3f* _v)
{   

    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector3f p(x, y, 0);
    Vector3f v0 = _v[1] - _v[0];
    Vector3f v1 = _v[2] - _v[1];
    Vector3f v2 = _v[0] - _v[2];

    Vector3f c0 = p - _v[0];
    Vector3f c1 = p - _v[1];
    Vector3f c2 = p - _v[2];

    float z0 = v0.cross(c0).z();
    float z1 = v1.cross(c1).z();
    float z2 = v2.cross(c2).z();

    // 判断同号
    return (z0 >= 0 && z1 >= 0 && z2 >= 0) || (z0 <= 0 && z1 <= 0 && z2 <= 0);
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
        downsampling();
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel 

    int x_min = std::min(std::min(v[0].x(), v[1].x()), v[2].x());
    int x_max = std::max(std::max(v[0].x(), v[1].x()), v[2].x());
    int y_min = std::min(std::min(v[0].y(), v[1].y()), v[2].y());
    int y_max = std::max(std::max(v[0].y(), v[1].y()), v[2].y());
    float ss_min_x = x_min * ssaa_factor;
    float ss_max_x = x_max * ssaa_factor;
    float ss_min_y = y_min * ssaa_factor;
    float ss_max_y = y_max * ssaa_factor;

    int ss_xmin = std::max(0, static_cast<int>(std::floor(ss_min_x)));
    int ss_xmax = std::min(width * ssaa_factor - 1, static_cast<int>(std::ceil(ss_max_x)));
    int ss_ymin = std::max(0, static_cast<int>(std::floor(ss_min_y)));
    int ss_ymax = std::min(height * ssaa_factor - 1, static_cast<int>(std::ceil(ss_max_y)));
    for (int y = ss_ymin; y <= ss_ymax; ++y) {
        for (int x = ss_xmin; x <= ss_xmax; ++x) {
            // 转换为原始坐标系的坐标 (0.5是样本点偏移量)
            float pixel_x = x / static_cast<float>(ssaa_factor) + 0.5f;
            float pixel_y = y / static_cast<float>(ssaa_factor) + 0.5f;

            // 检查当前超采样点是否在三角形内
            if (insideTriangle(pixel_x, pixel_y, t.v)) {
                auto [alpha, beta, gamma] = computeBarycentric2D(pixel_x, pixel_y, t.v);
                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                int ss_index = get_ssaa_index(x, y);

                // 深度测试
                if (z_inter< depth_buf_ssaa[ss_index]) {
                    // 更新超采样缓冲polated 区的颜色和深度
                    depth_buf_ssaa[ss_index] = z_interpolated;
                    frame_buf_ssaa[ss_index] = t.getColor();
                }
            }
        }
    }


    //     float px = x + 0.5f;
    //     float py = y + 0.5f;

    //     // 3️⃣ 判断是否在三角形内
    //     if (insideTriangle(px, py, t.v))
    //     {
    //         // 4️⃣ 计算插值深度
    //             auto[alpha, beta, gamma] = computeBarycentric2D(px, py, t.v);
    //             float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //             float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();

    //         // 5️⃣ 深度测试
    //         int index = get_index(x, y);
    //         if (z_interpolated < depth_buf[index])
    //         {
    //             // 6️⃣ 更新深度缓冲区
    //             depth_buf[index] = z_interpolated;

    //             // 7️⃣ 插值颜色
    //             Eigen::Vector3f color =
    //                 alpha * t.color[0] + beta * t.color[1] + gamma * t.color[2];

    //             // 8️⃣ 绘制像素
    //             set_pixel(Eigen::Vector3f(x, y, 1.0f), color * 255.0f);}
    // }}}
    // If so, use the following code to get the interpolated z value.
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(frame_buf_ssaa.begin(), frame_buf_ssaa.end(), Eigen::Vector3f{ 0, 0, 0 });
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    frame_buf_ssaa.resize(w * ssaa_factor * h * ssaa_factor);
    depth_buf_ssaa.resize(w * ssaa_factor * h * ssaa_factor);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

};
int rst::rasterizer::get_ssaa_index(int x, int y) {
    return (height * ssaa_factor - 1 - y) * width * ssaa_factor + x;
}//像素index是从上到下堆叠的顺序

void rst::rasterizer::downsampling() {
    if (ssaa_factor == 1) {
        // 不使用 SSAA，直接复制
        std::copy(frame_buf_ssaa.begin(), frame_buf_ssaa.end(), frame_buf.begin());
        return;
    }

    // 下采样过程
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Eigen::Vector3f avg_color(0, 0, 0);

            // 累加超采样像素
            for (int dy = 0; dy < ssaa_factor; ++dy) {
                for (int dx = 0; dx < ssaa_factor; ++dx) {
                    int ss_x = x * ssaa_factor + dx;
                    int ss_y = y * ssaa_factor + dy;

                    int ss_index = get_ssaa_index(ss_x, ss_y);
                    avg_color += frame_buf_ssaa[ss_index];
                }
            }

            // 计算平均值
            avg_color /= (ssaa_factor * ssaa_factor);

            // 设置到帧缓冲区
            int index = get_index(x, y);
            frame_buf[index] = avg_color;
        }
    }
}

// clang-format on