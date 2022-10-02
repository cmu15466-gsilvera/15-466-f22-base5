#include "WalkMesh.hpp"

#include "read_write_chunk.hpp"

#include <glm/gtx/norm.hpp>
#include <glm/gtx/string_cast.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

WalkMesh::WalkMesh(std::vector<glm::vec3> const& vertices_, std::vector<glm::vec3> const& normals_, std::vector<glm::uvec3> const& triangles_)
    : vertices(vertices_)
    , normals(normals_)
    , triangles(triangles_)
{

    // construct next_vertex map (maps each edge to the next vertex in the triangle):
    next_vertex.reserve(triangles.size() * 3);
    auto do_next = [this](uint32_t a, uint32_t b, uint32_t c) {
        auto ret = next_vertex.insert(std::make_pair(glm::uvec2(a, b), c));
        assert(ret.second);
    };
    for (auto const& tri : triangles) {
        do_next(tri.x, tri.y, tri.z);
        do_next(tri.y, tri.z, tri.x);
        do_next(tri.z, tri.x, tri.y);
    }

    // DEBUG: are vertex normals consistent with geometric normals?
    for (auto const& tri : triangles) {
        glm::vec3 const& a = vertices[tri.x];
        glm::vec3 const& b = vertices[tri.y];
        glm::vec3 const& c = vertices[tri.z];
        glm::vec3 out = glm::normalize(glm::cross(b - a, c - a));

        float da = glm::dot(out, normals[tri.x]);
        float db = glm::dot(out, normals[tri.y]);
        float dc = glm::dot(out, normals[tri.z]);

        assert(da > 0.1f && db > 0.1f && dc > 0.1f);
    }
}

glm::vec3 barycentric_weights(glm::vec3 const& a, glm::vec3 const& b, glm::vec3 const& c, glm::vec3 const& pt)
{
    // https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    // the coordinates of P with respect to triangle ABC are equivalent to the (signed) ratios of the
    // areas of PBC, PCA and PAB to the area of the reference triangle ABC

    // compute areas of smaller triangles formed with pt
    const glm::vec3 n = glm::normalize(glm::cross(c - a, b - c)); // normal of tri{a, b, c}
    const float PBC = glm::dot(glm::cross(b - pt, c - pt), n); // triangle formed by {b, c, pt}
    const float PCA = glm::dot(glm::cross(c - pt, a - pt), n); // triangle formed by {c, a, pt}
    const float PAB = glm::dot(glm::cross(a - pt, b - pt), n); // triangle formed by {a, b, pt}

    // divide by sum to account for constant scale factors and ensure weights sum to 1
    return glm::vec3(PBC, PCA, PAB) / (PBC + PCA + PAB);
}

WalkPoint WalkMesh::nearest_walk_point(glm::vec3 const& world_point) const
{
    assert(!triangles.empty() && "Cannot start on an empty walkmesh");

    WalkPoint closest;
    float closest_dis2 = std::numeric_limits<float>::infinity();

    for (auto const& tri : triangles) {
        // find closest point on triangle:

        glm::vec3 const& a = vertices[tri.x];
        glm::vec3 const& b = vertices[tri.y];
        glm::vec3 const& c = vertices[tri.z];

        // get barycentric coordinates of closest point in the plane of (a,b,c):
        glm::vec3 coords = barycentric_weights(a, b, c, world_point);

        // is that point inside the triangle?
        if (coords.x >= 0.0f && coords.y >= 0.0f && coords.z >= 0.0f) {
            // yes, point is inside triangle.
            float dis2 = glm::length2(world_point - to_world_point(WalkPoint(tri, coords)));
            if (dis2 < closest_dis2) {
                closest_dis2 = dis2;
                closest.indices = tri;
                closest.weights = coords;
            }
        } else {
            // check triangle vertices and edges:
            auto check_edge = [&world_point, &closest, &closest_dis2, this](uint32_t ai, uint32_t bi, uint32_t ci) {
                glm::vec3 const& a = vertices[ai];
                glm::vec3 const& b = vertices[bi];

                // find closest point on line segment ab:
                float along = glm::dot(world_point - a, b - a);
                float max = glm::dot(b - a, b - a);
                glm::vec3 pt;
                glm::vec3 coords;
                if (along < 0.0f) {
                    pt = a;
                    coords = glm::vec3(1.0f, 0.0f, 0.0f);
                } else if (along > max) {
                    pt = b;
                    coords = glm::vec3(0.0f, 1.0f, 0.0f);
                } else {
                    float amt = along / max;
                    pt = glm::mix(a, b, amt);
                    coords = glm::vec3(1.0f - amt, amt, 0.0f);
                }

                float dis2 = glm::length2(world_point - pt);
                if (dis2 < closest_dis2) {
                    closest_dis2 = dis2;
                    closest.indices = glm::uvec3(ai, bi, ci);
                    closest.weights = coords;
                }
            };
            check_edge(tri.x, tri.y, tri.z);
            check_edge(tri.y, tri.z, tri.x);
            check_edge(tri.z, tri.x, tri.y);
        }
    }
    assert(closest.indices.x < vertices.size());
    assert(closest.indices.y < vertices.size());
    assert(closest.indices.z < vertices.size());
    return closest;
}

void WalkMesh::walk_in_triangle(WalkPoint const& start, glm::vec3 const& step, WalkPoint* end_, float* time_) const
{
    assert(end_);
    auto& end = *end_;

    assert(time_);
    auto& time = *time_;

    const glm::vec3 world_start = to_world_point(start);
    const glm::vec3 world_point = world_start + step;

    glm::vec3 step_coords;
    glm::vec3 const& a = vertices[start.indices.x];
    glm::vec3 const& b = vertices[start.indices.y];
    glm::vec3 const& c = vertices[start.indices.z];
    { // project 'step' into a barycentric-coordinates direction:
        step_coords = barycentric_weights(a, b, c, world_point);
    }

    // if no edge is crossed, event will just be taking the whole step:
    {
        time = 1.0f;
        end = start;
        end.weights = step_coords;
    }

    // figure out which edge (if any) is crossed first.
    //  set time and end appropriately.

    // this condition ensures an edge was crossed
    if (!(step_coords.x >= 0.0f && step_coords.y >= 0.0f && step_coords.z >= 0.0f)) {
        // from: https://15466.courses.cs.cmu.edu/lesson/walkmesh
        // A particularly elegant way to write walk_in_triangle is to convert step into a "barycentric velocity", v.
        // This reduces the problem to checking when/if the motion of the original weights, w, under the velocity, v,
        // reaches a triangle edge. That is, checking for the earliest t that brings some weight in w + tv to zero.

        glm::vec3 barycentric_velocity;
        {
            // this is barycentric velocity bc velocity = X + t*V, and X = start.weights, then since the "end" weights
            // for end should be step_coords then end @ t == 1 should be still step_coords, therefore the "end" weights
            // should be start.weights + time * (velocity) => velocity = (step_coords - start.weights) so t == 1 => step_coords
            barycentric_velocity = step_coords - start.weights;
            // sum of barycentric velocity should equal 0 (sum(step_coords)==1 - sum(start.weights)==1)
            assert(std::fabs(barycentric_velocity.x + barycentric_velocity.y + barycentric_velocity.z) <= 0.001f);
        }

        // find the earliest time t (and associated component)
        int over_idx; // find the index of the velocity component that is over the triangle bounds
        {
            auto check_vel_component = [&time, &over_idx, &barycentric_velocity, &start](int component_idx) {
                if (barycentric_velocity[component_idx] < 0) { // crossing over in vel[component_idx] direction
                    // want t s.t. w + t * v == 0
                    // => t = -w / v
                    float t = -start.weights[component_idx] / barycentric_velocity[component_idx];
                    if (0 <= t && t < time && t < 1.f) {
                        time = t;
                        over_idx = component_idx;
                    }
                }
            };
            // check the x, y, z components of velocity to find the earliest t bringing w + tv == 0
            check_vel_component(0);
            check_vel_component(1);
            check_vel_component(2);
        }
        assert(0.f <= time && time <= 1.f);
        assert(0 <= over_idx && over_idx < 3); // can be only 0, 1, 2

        // the endpt maintains the same indices, and the weights correspond to the new barycentric position
        // but we'll need to clamp the z weight to 0 so its always on an edge
        end = WalkPoint(start.indices, start.weights + time * barycentric_velocity);

        // Remember: our convention is that when a WalkPoint is on an edge,
        // then wp.weights.z == 0.0f (so will likely need to re-order the indices)
        {
            if (over_idx == 0) {
                // velocity.x < 0 => crossing boundary in x axis (edge weight == 0)
                end.indices = glm::uvec3(end.indices.y, end.indices.z, end.indices.x);
                end.weights = glm::vec3(end.weights.y, end.weights.z, 0.f);
            } else if (over_idx == 1) {
                // velocity.y < 0 => crossing boundary in y axis (edge weight == 0)
                end.indices = glm::uvec3(end.indices.z, end.indices.x, end.indices.y);
                end.weights = glm::vec3(end.weights.z, end.weights.x, 0.f);
            } else if (over_idx == 2) {
                // velocity.z < 0 => crossing boundary in z axis (edge weight == 0)
                end.indices = glm::uvec3(end.indices.x, end.indices.y, end.indices.z);
                end.weights = glm::vec3(end.weights.x, end.weights.y, 0.f);
            } else {
                throw std::runtime_error("Unable to re-order indices!");
            }
        }
    }
}

bool WalkMesh::cross_edge(WalkPoint const& start, WalkPoint* end_, glm::quat* rotation_) const
{
    assert(end_);
    auto& end = *end_;

    assert(rotation_);
    auto& rotation = *rotation_;

    assert(start.indices.x <= vertices.size() && start.indices.y <= vertices.size() && start.indices.z <= vertices.size());
    assert(start.weights.z == 0.0f); //*must* be on an edge.
    const glm::uvec2 edge = glm::uvec2(start.indices.y, start.indices.x); // looking for opposite vertex to this edge

    // check if 'edge' is a non-boundary edge:
    if (next_vertex.find(edge) != next_vertex.end()) { // found in next_vertex (opposite vertex exists in mesh)
        const uint32_t other_pt = next_vertex.find(edge)->second; // this should always work since != end()

        // make 'end' represent the same (world) point, but on triangle (edge.y, edge.x, [other point]):
        end = WalkPoint(
            glm::vec3(start.indices.y, start.indices.x, other_pt), // new indices
            glm::vec3(start.weights.y, start.weights.x, start.weights.z) // new weights
        );

        // make 'rotation' the rotation that takes (start.indices)'s normal to (end.indices)'s normal:
        const glm::vec3& A = vertices[start.indices.x];
        const glm::vec3& B = vertices[start.indices.y];
        const glm::vec3& C = vertices[start.indices.z];
        const glm::vec3& D = vertices[other_pt]; // next (next) edge

        const glm::vec3 n0 = glm::normalize(glm::cross(A - B, B - C)); // normal 0 (starting)
        const glm::vec3 n1 = glm::normalize(glm::cross(B - A, A - D)); // normal 1 (ending)
        rotation = glm::rotation(n0, n1);

        return true;
    }

    end = start;
    rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    return false;
}

WalkMeshes::WalkMeshes(std::string const& filename)
{
    std::ifstream file(filename, std::ios::binary);

    std::vector<glm::vec3> vertices;
    read_chunk(file, "p...", &vertices);

    std::vector<glm::vec3> normals;
    read_chunk(file, "n...", &normals);

    std::vector<glm::uvec3> triangles;
    read_chunk(file, "tri0", &triangles);

    std::vector<char> names;
    read_chunk(file, "str0", &names);

    struct IndexEntry {
        uint32_t name_begin, name_end;
        uint32_t vertex_begin, vertex_end;
        uint32_t triangle_begin, triangle_end;
    };

    std::vector<IndexEntry> index;
    read_chunk(file, "idxA", &index);

    if (file.peek() != EOF) {
        std::cerr << "WARNING: trailing data in walkmesh file '" << filename << "'" << std::endl;
    }

    //-----------------

    if (vertices.size() != normals.size()) {
        throw std::runtime_error("Mis-matched position and normal sizes in '" + filename + "'");
    }

    for (auto const& e : index) {
        if (!(e.name_begin <= e.name_end && e.name_end <= names.size())) {
            throw std::runtime_error("Invalid name indices in index of '" + filename + "'");
        }
        if (!(e.vertex_begin <= e.vertex_end && e.vertex_end <= vertices.size())) {
            throw std::runtime_error("Invalid vertex indices in index of '" + filename + "'");
        }
        if (!(e.triangle_begin <= e.triangle_end && e.triangle_end <= triangles.size())) {
            throw std::runtime_error("Invalid triangle indices in index of '" + filename + "'");
        }

        // copy vertices/normals:
        std::vector<glm::vec3> wm_vertices(vertices.begin() + e.vertex_begin, vertices.begin() + e.vertex_end);
        std::vector<glm::vec3> wm_normals(normals.begin() + e.vertex_begin, normals.begin() + e.vertex_end);

        // remap triangles:
        std::vector<glm::uvec3> wm_triangles;
        wm_triangles.reserve(e.triangle_end - e.triangle_begin);
        for (uint32_t ti = e.triangle_begin; ti != e.triangle_end; ++ti) {
            if (!((e.vertex_begin <= triangles[ti].x && triangles[ti].x < e.vertex_end)
                    && (e.vertex_begin <= triangles[ti].y && triangles[ti].y < e.vertex_end)
                    && (e.vertex_begin <= triangles[ti].z && triangles[ti].z < e.vertex_end))) {
                throw std::runtime_error("Invalid triangle in '" + filename + "'");
            }
            wm_triangles.emplace_back(
                triangles[ti].x - e.vertex_begin,
                triangles[ti].y - e.vertex_begin,
                triangles[ti].z - e.vertex_begin);
        }

        std::string name(names.begin() + e.name_begin, names.begin() + e.name_end);

        auto ret = meshes.emplace(name, WalkMesh(wm_vertices, wm_normals, wm_triangles));
        if (!ret.second) {
            throw std::runtime_error("WalkMesh with duplicated name '" + name + "' in '" + filename + "'");
        }
    }
}

WalkMesh const& WalkMeshes::lookup(std::string const& name) const
{
    auto f = meshes.find(name);
    if (f == meshes.end()) {
        throw std::runtime_error("WalkMesh with name '" + name + "' not found.");
    }
    return f->second;
}
