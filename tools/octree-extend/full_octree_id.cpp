#include <array>
#include <cassert>
#include <string>
#include <utility>

constexpr auto NDIM = 3;

size_t to_id(std::array<int, NDIM> indices, int levels)
{
    size_t id = 1;
    for (int l = 0; l < levels; ++l)
    {
        for (int d = 0; d < NDIM; ++d)
        {
            id <<= 1;
            id |= ((indices[d] >> l) & 1);
        }
    }
    return id;
}

std::pair<std::array<int, NDIM>, int> parse_id(size_t id)
{
    std::array<int, NDIM> indices;
    indices.fill(0);

    int level = 0;

    for (; id != 1; ++level)
    {
        printf("%i %llo\n", level, id);
        for (int d = NDIM - 1; d >= 0; d--)
        {
            indices[d] <<= 1;
            indices[d] |= (id & 1);
            id >>= 1;
        }
    }
    return std::make_pair(std::move(indices), level);
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("Usage: full_octree <string sub-grid id> <number of additional "
               "levels>\n");
        return 1;
    }

    size_t subgrid_id = std::stoull(argv[1], nullptr, 8);
    int additional_levels = std::atoi(argv[2]);

    std::array<int, NDIM> subgrid_indices;
    int base_level;
    std::tie(subgrid_indices, base_level) = parse_id(subgrid_id);

    printf("base_level is %i\n", base_level);
    for (int i = 0; i < (1 << additional_levels); ++i)
    {
        for (int j = 0; j < (1 << additional_levels); ++j)
        {
            for (int k = 0; k < (1 << additional_levels); ++k)
            {
                std::array<int, NDIM> const indices{
                    (subgrid_indices[0] << additional_levels) + i,
                    (subgrid_indices[1] << additional_levels) + j,
                    (subgrid_indices[2] << additional_levels) + k};

                size_t new_id = to_id(indices, base_level + additional_levels);
                printf("the (%i,%i,%i) cell in sub-grid %s has full octree id "
                       "%o\n",
                    i, j, k, argv[1], new_id);
            }
        }
    }
    return 0;
}
