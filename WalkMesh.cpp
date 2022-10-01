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
    const glm::vec3 h = glm::cross(c - a, b - c); // entire triangle formed by {a, b, c}
    const float BCP = glm::dot(glm::cross(b - pt, c - pt), h); // triangle formed by {b, c, pt}
    const float ACP = glm::dot(glm::cross(c - pt, a - pt), h); // triangle formed by {a, c, pt}
    const float ABP = glm::dot(glm::cross(a - pt, b - pt), h); // triangle formed by {a, b, pt}

    // divide by sum to account for constant scale factors and ensure weights sum to 1
    return glm::vec3(BCP, ACP, ABP) / (BCP + ACP + ABP);
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

    if (!(step_coords.x >= 0.0f && step_coords.y >= 0.0f && step_coords.z >= 0.0f)) {

        // figure out which edge (if any) is crossed first.
        //  set time and end appropriately.
        // check triangle vertices and edges:
        float closest_dis2 = std::numeric_limits<float>::infinity(); // to only update end to closest edge-pt
        auto check_edge = [&world_point, &time, &world_start, &end, &closest_dis2, this](uint32_t ai, uint32_t bi, uint32_t ci) {
            glm::vec3 const& a = vertices[ai];
            glm::vec3 const& b = vertices[bi];
            glm::vec3 const& c = vertices[ci];

            // find closest point on line segment ab:
            glm::vec3 pt;
            glm::vec3 coords;

            const glm::vec3 norm = glm::cross(b - a, c - a); // normal of triangle
            const glm::vec3 world_dir = world_point - world_start;
            auto proj = [](const glm::vec3& u, const glm::vec3& v) -> glm::vec3 {
                return (glm::dot(u, v) / glm::length2(v)) * v;
            };
            // project world_dir onto triangle (subtract component in normal dir)
            const glm::vec3 world_dir_proj = world_dir - proj(world_dir, norm);
            const glm::vec3 world_end = world_start + world_dir_proj;

            // early out test if the line segments don't cross
            {
                // the walk_point + step does not cross the edge formed by a, b
                if (glm::dot(glm::cross(b - a, world_end - a), glm::cross(b - a, world_start - a)) > 0)
                    return; // early out for edges that don't get crossed
            }

            // compute intersection point
            {
                // Recall that a 'line' can be defined as (L = origin(0) + t * direction(Dir)) for some t

                // Calculating shortest line segment intersecting both lines
                // Implementation sourced from http://paulbourke.net/geometry/pointlineplane/
                const glm::vec3& L0 = world_start;
                const glm::vec3& LDir = world_dir_proj;
                const glm::vec3& R0 = a;
                const glm::vec3 RDir = (b - a);

                const glm::vec3 L0R0 = L0 - R0; // segment between L origin and R origin

                // Calculating dot-product equation to find perpendicular shortest-line-segment
                const float d1343 = L0R0.x * RDir.x + L0R0.y * RDir.y + L0R0.z * RDir.z;
                const float d4321 = RDir.x * LDir.x + RDir.y * LDir.y + RDir.z * LDir.z;
                const float d1321 = L0R0.x * LDir.x + L0R0.y * LDir.y + L0R0.z * LDir.z;
                const float d4343 = RDir.x * RDir.x + RDir.y * RDir.y + RDir.z * RDir.z;
                const float d2121 = LDir.x * LDir.x + LDir.y * LDir.y + LDir.z * LDir.z;
                const float denom = d2121 * d4343 - d4321 * d4321;
                if (std::fabs(denom) < 0.00001f) // else no intersection
                    return;
                const float numer = d1343 * d4321 - d1321 * d4343;

                // calculate scalars (mu) that scale the unit direction XDir to reach the desired points
                const float muL = numer / denom; // variable scale of direction vector for LEFT ray
                const float muR = (d1343 + d4321 * (muL)) / d4343; // variable scale of direction vector for RIGHT ray

                // calculate the points on the respective rays that create the intersecting line
                const glm::vec3 ptL = L0 + muL * LDir; // the point on the Left ray
                const glm::vec3 ptR = R0 + muR * RDir; // the point on the Right ray

                // calculate the vector between the middle of the two endpoints and return its magnitude
                pt = (ptL + ptR) / 2.0f; // middle point between two endpoints of shortest-line-segment
            }

            // compute the weights
            {
                // Remember: our convention is that when a WalkPoint is on an edge,
                //  then wp.weights.z == 0.0f (so will likely need to re-order the indices)
                float amt = glm::length2(pt - a) / glm::length2(b - a);
                if (amt < 0.f || amt > 1.f)
                    return; // early out for invalid barycentric coords
                coords = glm::vec3(1.0f - amt, amt, 0.0f);
            }

            // update end/time appropriately
            {
                float dis2 = glm::dot(pt - world_start, world_end - world_start); // smallest dot
                if (dis2 < closest_dis2) {
                    closest_dis2 = dis2;
                    time = glm::length(pt - world_start) / glm::length(world_point - world_start);
                    // end is the point on the edge
                    end.indices = glm::uvec3(ai, bi, ci);
                    end.weights = coords;
                }
            }
        };

        check_edge(start.indices.x, start.indices.y, start.indices.z);
        check_edge(start.indices.y, start.indices.z, start.indices.x);
        check_edge(start.indices.z, start.indices.x, start.indices.y);
    }
}

bool WalkMesh::cross_edge(WalkPoint const& start, WalkPoint* end_, glm::quat* rotation_) const
{
    assert(end_);
    auto& end = *end_;

    assert(rotation_);
    auto& rotation = *rotation_;

    std::cout << "weights: " << glm::to_string(start.weights) << std::endl;
    assert(start.weights.z == 0.0f); //*must* be on an edge.
    glm::uvec2 edge = glm::uvec2(start.indices);

    // check if 'edge' is a non-boundary edge:
    edge = glm::uvec2(edge.y, edge.x); // why do this?
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
    } else {
        end = start;
        rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
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
